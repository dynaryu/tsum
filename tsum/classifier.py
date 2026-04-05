"""
Monotone classifier for system reliability estimation via importance sampling.

Instead of enumerating reference states (which cover negligible probability in
high dimensions), this module:

1. Trains a monotone classifier f(x) ~ Phi(x) from (component-state, system-state)
   samples obtained by evaluating the true system function.
2. Uses the classifier to design an importance sampling distribution that
   concentrates samples in the failure region.
3. Estimates P(failure) via importance sampling with the TRUE system function,
   giving an unbiased estimate regardless of classifier accuracy.

The classifier respects coherence (monotonicity): improving any component never
worsens the system.  This is enforced via monotone_constraints in the gradient-
boosted tree model.
"""

import time
import json
import os
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ── Model wrapper ─────────────────────────────────────────────────────────────

class MonotoneClassifier:
    """
    Gradient-boosted tree classifier with per-feature monotone constraints.

    For a coherent system where higher component state => better system state,
    all features get monotone_constraints = +1.

    Features are integer component-state indices (compact, not one-hot).
    Target is binary: 0 = failure (S <= threshold), 1 = survival (S > threshold).
    """

    def __init__(self, n_vars: int, n_state: int):
        self.n_vars = n_vars
        self.n_state = n_state
        self._clf = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "MonotoneClassifier":
        """
        Train the classifier.

        Args:
            X: (n_samples, n_vars) integer component-state indices
            y: (n_samples,) binary labels: 0 = failure, 1 = survival
        """
        from sklearn.ensemble import HistGradientBoostingClassifier

        # All features are monotone increasing for a coherent system:
        # higher state index => better component => system at least as good
        mono = [1] * self.n_vars

        default_min_leaf = min(20, max(1, len(X) // 4))
        self._clf = HistGradientBoostingClassifier(
            max_iter=kwargs.get("max_iter", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.05),
            min_samples_leaf=kwargs.get("min_samples_leaf", default_min_leaf),
            monotonic_cst=mono,
            random_state=42,
        )
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba_failure(self, X: np.ndarray) -> np.ndarray:
        """Return P(failure) for each sample, shape (n_samples,)."""
        if not self._fitted:
            return np.full(X.shape[0], 0.5)
        proba = self._clf.predict_proba(X)
        classes = list(self._clf.classes_)
        fail_idx = classes.index(0) if 0 in classes else 0
        return proba[:, fail_idx].astype(np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class (0 or 1) for each sample."""
        if not self._fitted:
            return np.ones(X.shape[0], dtype=int)
        return self._clf.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self._fitted:
            return 0.0
        return self._clf.score(X, y)


# ── Data generation ───────────────────────────────────────────────────────────

def sample_component_states(
    probs_dict: Dict[str, Dict],
    row_names: List[str],
    n_samples: int,
    n_state: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample component-state vectors from the joint distribution (independent).

    Returns:
        X: (n_samples, n_vars) integer array of component state indices
    """
    n_vars = len(row_names)
    X = np.empty((n_samples, n_vars), dtype=np.int32)
    for i, name in enumerate(row_names):
        p = probs_dict[name]
        probs = [p[str(s)]["p"] if str(s) in p else 0.0 for s in range(n_state)]
        # Normalize (handles padding for components with fewer states)
        total = sum(probs)
        if total > 0:
            probs = [pp / total for pp in probs]
        X[:, i] = rng.choice(n_state, size=n_samples, p=probs)
    return X


_worker_sfun = None
_worker_X = None
_worker_row_names = None


def _worker_init(sfun, X, row_names):
    """Initializer for Pool workers — stores unpicklable objects as globals."""
    global _worker_sfun, _worker_X, _worker_row_names
    _worker_sfun = sfun
    _worker_X = X
    _worker_row_names = row_names


def _worker_eval(i):
    """Evaluate sfun for sample i using worker-global state."""
    comps_st = {_worker_row_names[k]: int(_worker_X[i, k])
                for k in range(len(_worker_row_names))}
    _, sys_st, _ = _worker_sfun(comps_st)
    return sys_st


def evaluate_sfun_batch(
    X: np.ndarray,
    sfun: Callable,
    row_names: List[str],
    n_workers: int = 1,
) -> np.ndarray:
    """
    Evaluate the system function for a batch of component-state vectors.

    Args:
        X: (n_samples, n_vars) integer component-state indices
        sfun: system function callable
        row_names: component names
        n_workers: number of parallel workers

    Returns:
        y: (n_samples,) integer system states
    """
    n_samples = X.shape[0]

    if n_workers > 1:
        from multiprocessing import Pool
        with Pool(n_workers, initializer=_worker_init,
                  initargs=(sfun, X, row_names)) as pool:
            y = pool.map(_worker_eval, range(n_samples))
        return np.array(y, dtype=np.int32)
    else:
        y = np.empty(n_samples, dtype=np.int32)
        for i in range(n_samples):
            comps_st = {row_names[k]: int(X[i, k])
                        for k in range(len(row_names))}
            _, sys_st, _ = sfun(comps_st)
            y[i] = sys_st
        return y


# ── Importance sampling ───────────────────────────────────────────────────────

def build_is_distribution(
    probs_dict: Dict[str, Dict],
    row_names: List[str],
    n_state: int,
    classifier: MonotoneClassifier,
    *,
    n_pilot: int = 100_000,
    shift_factor: float = 2.0,
    rng: np.random.Generator,
) -> List[List[float]]:
    """
    Build an importance sampling distribution that concentrates mass on the
    failure region, guided by the classifier.

    Strategy: for components the classifier deems important for failure,
    shift probability mass toward degraded states.  The shift is proportional
    to the classifier's feature importance (split-count based).

    Args:
        probs_dict: original component probabilities
        row_names: component names
        n_state: max number of states
        classifier: trained MonotoneClassifier
        n_pilot: number of pilot samples for calibration
        shift_factor: controls aggressiveness of the shift (higher = more bias)
        rng: numpy random generator

    Returns:
        is_probs: list of per-component probability vectors for IS
    """
    n_vars = len(row_names)

    # Get feature importance from the classifier
    importance = np.zeros(n_vars)
    if classifier._fitted and hasattr(classifier._clf, '_predictors'):
        for predictors_at_iter in classifier._clf._predictors:
            for predictor in predictors_at_iter:
                nodes = predictor.nodes
                if hasattr(nodes, 'dtype') and 'feature_idx' in nodes.dtype.names:
                    for node in nodes:
                        is_leaf = bool(node['is_leaf']) if 'is_leaf' in nodes.dtype.names else False
                        if not is_leaf:
                            fi = int(node['feature_idx'])
                            if 0 <= fi < n_vars:
                                importance[fi] += 1
        total = importance.sum()
        if total > 0:
            importance /= total

    # Build shifted distributions: for important components, shift mass
    # toward lower (degraded) states
    is_probs = []
    for i, name in enumerate(row_names):
        p = probs_dict[name]
        orig = np.array([p[str(s)]["p"] if str(s) in p else 0.0
                         for s in range(n_state)], dtype=np.float64)
        total = orig.sum()
        if total > 0:
            orig /= total

        # Shift factor scales with component importance.
        # When importance is all zeros (no failures seen), apply uniform shift
        # so IS still explores the failure region.
        if importance.max() > 0:
            comp_shift = shift_factor * (importance[i] / importance.max())
        else:
            comp_shift = shift_factor

        # Apply shift: multiply P(state=s) by exp(comp_shift * (max_state - s))
        # This increases probability of lower (worse) states
        max_s = len(p) - 1
        weights = np.array([np.exp(comp_shift * (max_s - s)) for s in range(n_state)])
        shifted = orig * weights
        shifted_sum = shifted.sum()
        if shifted_sum > 0:
            shifted /= shifted_sum
        else:
            shifted = orig.copy()

        # Ensure no zero probabilities (for IS weight computation)
        shifted = np.clip(shifted, 1e-10, None)
        shifted /= shifted.sum()

        is_probs.append(shifted.tolist())

    return is_probs


def sample_is(
    is_probs: List[List[float]],
    n_samples: int,
    n_state: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from the importance sampling distribution."""
    n_vars = len(is_probs)
    X = np.empty((n_samples, n_vars), dtype=np.int32)
    for i in range(n_vars):
        X[:, i] = rng.choice(n_state, size=n_samples, p=is_probs[i])
    return X


def compute_is_weights(
    X: np.ndarray,
    probs_dict: Dict[str, Dict],
    is_probs: List[List[float]],
    row_names: List[str],
    n_state: int,
) -> np.ndarray:
    """
    Compute importance sampling likelihood ratios: w(x) = p(x) / q(x).

    Args:
        X: (n_samples, n_vars) sampled component states
        probs_dict: original probabilities p
        is_probs: IS probabilities q

    Returns:
        weights: (n_samples,) likelihood ratios
    """
    n_samples, n_vars = X.shape

    # Precompute original probs as array
    orig_probs = np.empty((n_vars, n_state), dtype=np.float64)
    for i, name in enumerate(row_names):
        p = probs_dict[name]
        for s in range(n_state):
            orig_probs[i, s] = p[str(s)]["p"] if str(s) in p else 0.0
        total = orig_probs[i].sum()
        if total > 0:
            orig_probs[i] /= total

    is_arr = np.array(is_probs, dtype=np.float64)  # (n_vars, n_state)

    # Compute log-likelihood ratios for numerical stability
    log_weights = np.zeros(n_samples, dtype=np.float64)
    for j in range(n_vars):
        states_j = X[:, j]
        log_p = np.log(orig_probs[j, states_j] + 1e-300)
        log_q = np.log(is_arr[j, states_j] + 1e-300)
        log_weights += log_p - log_q

    return np.exp(log_weights)


# ── Active learning ───────────────────────────────────────────────────────────

def select_active_samples(
    probs_dict: Dict[str, Dict],
    row_names: List[str],
    n_state: int,
    classifier: MonotoneClassifier,
    n_candidates: int,
    n_select: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select samples near the classifier's decision boundary for active learning.

    Samples where P(failure) ~ 0.5 are most informative for refining the
    boundary.  We also include some samples biased toward degradation to
    explore the failure region.

    Returns:
        X_selected: (n_select, n_vars) component-state vectors to evaluate
    """
    # Generate candidates from the original distribution
    X_cand = sample_component_states(
        probs_dict, row_names, n_candidates, n_state, rng)

    # Score with classifier
    p_fail = classifier.predict_proba_failure(X_cand)

    # Uncertainty = closeness to 0.5
    uncertainty = 1.0 - 2.0 * np.abs(p_fail - 0.5)

    # Also weight by failure probability to explore failure region
    score = uncertainty + 0.5 * p_fail

    # Select top-scoring candidates
    n_select = min(n_select, len(X_cand))
    top_idx = np.argsort(score)[-n_select:]
    return X_cand[top_idx]


# ── Main estimation pipeline ──────────────────────────────────────────────────

@dataclass
class EstimationResult:
    """Result of the classifier-based probability estimation."""
    p_failure: float = 0.0
    p_failure_se: float = 0.0             # standard error
    p_failure_ci_lower: float = 0.0       # 95% CI
    p_failure_ci_upper: float = 0.0
    n_is_samples: int = 0
    n_training_samples: int = 0
    n_failures_observed: int = 0
    classifier_accuracy: float = 0.0
    rounds: List[Dict[str, Any]] = field(default_factory=list)


def estimate_failure_probability(
    sfun: Callable,
    probs_dict: Dict[str, Dict],
    row_names: List[str],
    n_state: int,
    *,
    # Training
    n_initial_samples: int = 5_000,
    n_active_rounds: int = 5,
    n_active_samples_per_round: int = 2_000,
    n_active_candidates: int = 50_000,
    # Importance sampling
    n_is_samples: int = 100_000,
    is_shift_factor: float = 3.0,
    # Computation
    n_workers: int = 1,
    seed: int = 42,
    # Output
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> EstimationResult:
    """
    Estimate P(failure) using a monotone classifier and importance sampling.

    Pipeline:
        1. Generate initial training data by sampling from component distribution
           and evaluating the true system function.
        2. Train monotone classifier.
        3. Active learning: iteratively select boundary samples, evaluate,
           retrain.
        4. Build importance sampling distribution from trained classifier.
        5. Draw IS samples, evaluate true Phi, compute unbiased estimate.

    Args:
        sfun: system function (comps_st -> (fval, sys_st, info))
        probs_dict: component probability distributions
        row_names: component names
        n_state: max states per component
        n_initial_samples: initial training set size
        n_active_rounds: number of active learning rounds
        n_active_samples_per_round: samples to evaluate per active round
        n_active_candidates: candidates to score for active selection
        n_is_samples: importance sampling samples for final estimate
        is_shift_factor: aggressiveness of IS distribution shift
        n_workers: parallel workers for sfun evaluation
        seed: random seed
        output_dir: directory to save results (None = don't save)
        verbose: print progress

    Returns:
        EstimationResult with P(failure) estimate and confidence interval
    """
    rng = np.random.default_rng(seed)
    n_vars = len(row_names)
    result = EstimationResult()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    # ── Phase 1: Initial training data ────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"Phase 1: Generate initial training data ({n_initial_samples} samples)")
    _log(f"{'='*60}")

    t0 = time.time()
    X_train = sample_component_states(
        probs_dict, row_names, n_initial_samples, n_state, rng)
    y_train = evaluate_sfun_batch(X_train, sfun, row_names, n_workers)
    y_binary = (y_train < 1).astype(np.int32)  # 0=survival, remap: 0=failure
    # Correct: y_binary[i] = 1 if system failed (sys_st < 1), 0 if survived
    # Actually: sfun returns sys_st=0 for failure, sys_st=1 for survival
    # We want: label 0 = failure, label 1 = survival
    y_binary = y_train.copy()  # sys_st directly: 0=failure, 1=survival

    n_fail_init = int((y_binary == 0).sum())
    elapsed = time.time() - t0
    _log(f"  Time: {elapsed:.1f}s")
    _log(f"  Failures found: {n_fail_init}/{n_initial_samples} "
         f"({n_fail_init/n_initial_samples:.4%})")

    result.rounds.append({
        "phase": "initial",
        "n_samples": n_initial_samples,
        "n_failures": n_fail_init,
        "time_sec": elapsed,
    })

    # ── Phase 2: Train classifier ─────────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"Phase 2: Train monotone classifier")
    _log(f"{'='*60}")

    classifier = MonotoneClassifier(n_vars, n_state)

    t0 = time.time()
    classifier.fit(X_train, y_binary)
    elapsed = time.time() - t0

    acc = classifier.score(X_train, y_binary)
    _log(f"  Training accuracy: {acc:.4f}")
    _log(f"  Training time: {elapsed:.1f}s")
    result.classifier_accuracy = acc

    # ── Phase 3: Active learning ──────────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"Phase 3: Active learning ({n_active_rounds} rounds)")
    _log(f"{'='*60}")

    total_active_samples = 0
    for r in range(n_active_rounds):
        t0 = time.time()

        # Select informative samples near the boundary
        X_active = select_active_samples(
            probs_dict, row_names, n_state, classifier,
            n_candidates=n_active_candidates,
            n_select=n_active_samples_per_round,
            rng=rng,
        )

        # Evaluate true system function
        y_active = evaluate_sfun_batch(X_active, sfun, row_names, n_workers)

        # Add to training set and retrain
        X_train = np.concatenate([X_train, X_active], axis=0)
        y_binary = np.concatenate([y_binary, y_active], axis=0)

        classifier.fit(X_train, y_binary)
        acc = classifier.score(X_train, y_binary)

        n_fail_active = int((y_active == 0).sum())
        total_active_samples += len(X_active)
        elapsed = time.time() - t0

        _log(f"  Round {r+1}/{n_active_rounds}: "
             f"+{len(X_active)} samples ({n_fail_active} failures), "
             f"accuracy={acc:.4f}, time={elapsed:.1f}s")

        result.rounds.append({
            "phase": "active",
            "round": r + 1,
            "n_samples": len(X_active),
            "n_failures": n_fail_active,
            "accuracy": acc,
            "time_sec": elapsed,
        })

    total_training = n_initial_samples + total_active_samples
    total_failures = int((y_binary == 0).sum())
    result.n_training_samples = total_training
    result.classifier_accuracy = acc
    _log(f"\n  Total training: {total_training} samples, "
         f"{total_failures} failures ({total_failures/total_training:.4%})")

    # ── Phase 4: Importance sampling ──────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"Phase 4: Importance sampling ({n_is_samples} samples)")
    _log(f"{'='*60}")

    t0 = time.time()

    # Build IS distribution
    is_probs = build_is_distribution(
        probs_dict, row_names, n_state, classifier,
        shift_factor=is_shift_factor, rng=rng,
    )

    # Sample from IS distribution
    X_is = sample_is(is_probs, n_is_samples, n_state, rng)

    # Evaluate TRUE system function (this is what makes the estimate unbiased)
    _log(f"  Evaluating {n_is_samples} IS samples with true system function...")
    y_is = evaluate_sfun_batch(X_is, sfun, row_names, n_workers)

    # Compute IS weights
    weights = compute_is_weights(X_is, probs_dict, is_probs, row_names, n_state)

    # Estimate P(failure) = E_q[w(x) * I(failure)]
    failure_indicator = (y_is == 0).astype(np.float64)
    weighted_failures = weights * failure_indicator

    p_fail_is = weighted_failures.mean()
    n_eff = (weights.sum() ** 2) / (weights ** 2).sum()

    # Standard error via CLT
    if n_is_samples > 1:
        se = np.sqrt(np.var(weighted_failures, ddof=1) / n_is_samples)
    else:
        se = float('inf')

    elapsed = time.time() - t0

    n_is_failures = int(failure_indicator.sum())
    _log(f"  IS failures observed: {n_is_failures}/{n_is_samples}")
    _log(f"  Effective sample size: {n_eff:.0f}")
    _log(f"  Time: {elapsed:.1f}s")

    result.p_failure = float(p_fail_is)
    result.p_failure_se = float(se)
    result.p_failure_ci_lower = float(max(0, p_fail_is - 1.96 * se))
    result.p_failure_ci_upper = float(p_fail_is + 1.96 * se)
    result.n_is_samples = n_is_samples
    result.n_failures_observed = n_is_failures
    result.rounds.append({
        "phase": "importance_sampling",
        "n_samples": n_is_samples,
        "n_failures": n_is_failures,
        "n_eff": float(n_eff),
        "p_failure": float(p_fail_is),
        "se": float(se),
        "time_sec": elapsed,
    })

    # ── Summary ───────────────────────────────────────────────────────────
    total_sfun_calls = total_training + n_is_samples
    _log(f"\n{'='*60}")
    _log(f"Results")
    _log(f"{'='*60}")
    _log(f"  P(failure)  = {p_fail_is:.6e}")
    _log(f"  SE          = {se:.6e}")
    _log(f"  95% CI      = [{result.p_failure_ci_lower:.6e}, "
         f"{result.p_failure_ci_upper:.6e}]")
    _log(f"  Total sfun calls: {total_sfun_calls}")
    _log(f"  Classifier accuracy: {result.classifier_accuracy:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    if output_dir:
        out = {
            "p_failure": result.p_failure,
            "p_failure_se": result.p_failure_se,
            "p_failure_ci_lower": result.p_failure_ci_lower,
            "p_failure_ci_upper": result.p_failure_ci_upper,
            "n_is_samples": result.n_is_samples,
            "n_training_samples": result.n_training_samples,
            "n_failures_observed": result.n_failures_observed,
            "classifier_accuracy": result.classifier_accuracy,
            "rounds": result.rounds,
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(out, f, indent=2)
        _log(f"\n  Results saved to {output_dir}/results.json")

    return result
