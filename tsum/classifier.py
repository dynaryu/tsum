"""
Monotone classifier for guided boundary search in TSUM rule extraction.

Provides a monotone gradient-boosted tree classifier that learns the
survival/failure boundary from system function evaluations.  Used within
``run_rule_extraction_by_mcs`` to bias sampling toward the decision boundary
where "unknown" states (not yet covered by any rule) are concentrated.

The classifier respects coherence (monotonicity): improving any component never
worsens the system.  This is enforced via monotone_constraints in the
gradient-boosted tree model.
"""

import numpy as np
import torch
from typing import Callable, Dict, List, Optional


# ── Model wrapper ─────────────────────────────────────────────────────────────

class MonotoneClassifier:
    """
    Gradient-boosted tree classifier with per-feature monotone constraints.

    For a coherent system where higher component state => better system state,
    all features get monotone_constraints = +1.

    Features are integer component-state indices (compact, not one-hot).
    Target is binary: 0 = failure, 1 = survival.
    """

    def __init__(self, n_vars: int, n_state: int):
        self.n_vars = n_vars
        self.n_state = n_state
        self._clf = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "MonotoneClassifier":
        from sklearn.ensemble import HistGradientBoostingClassifier

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


# ── Sampling helpers ─────────────────────────────────────────────────────────

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
        total = sum(probs)
        if total > 0:
            probs = [pp / total for pp in probs]
        X[:, i] = rng.choice(n_state, size=n_samples, p=probs)
    return X


def _indices_to_onehot(X_int: np.ndarray, n_state: int,
                       device: torch.device) -> torch.Tensor:
    """Convert (n_samples, n_vars) int indices to (n_samples, n_vars, n_state) one-hot."""
    X_t = torch.from_numpy(X_int).long().to(device)
    return torch.nn.functional.one_hot(X_t, num_classes=n_state).int()


# ── sfun evaluation (multiprocessing-safe) ───────────────────────────────────

_worker_sfun = None
_worker_X = None
_worker_row_names = None


def _worker_init(sfun, X, row_names):
    global _worker_sfun, _worker_X, _worker_row_names
    _worker_sfun = sfun
    _worker_X = X
    _worker_row_names = row_names


def _worker_eval(i):
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


# ── IS weight computation ────────────────────────────────────────────────────

def compute_is_weights(
    X: np.ndarray,
    probs_dict: Dict[str, Dict],
    is_probs: List[List[float]],
    row_names: List[str],
    n_state: int,
) -> np.ndarray:
    """
    Compute importance sampling likelihood ratios: w(x) = p(x) / q(x).

    Returns:
        weights: (n_samples,) likelihood ratios
    """
    n_samples, n_vars = X.shape

    orig_probs = np.empty((n_vars, n_state), dtype=np.float64)
    for i, name in enumerate(row_names):
        p = probs_dict[name]
        for s in range(n_state):
            orig_probs[i, s] = p[str(s)]["p"] if str(s) in p else 0.0
        total = orig_probs[i].sum()
        if total > 0:
            orig_probs[i] /= total

    is_arr = np.array(is_probs, dtype=np.float64)

    log_weights = np.zeros(n_samples, dtype=np.float64)
    for j in range(n_vars):
        states_j = X[:, j]
        log_p = np.log(orig_probs[j, states_j] + 1e-300)
        log_q = np.log(is_arr[j, states_j] + 1e-300)
        log_weights += log_p - log_q

    return np.exp(log_weights)


# ── Boundary-guided sampling for TSUM integration ───────────────────────────

def build_boundary_distribution(
    probs: torch.Tensor,
    classifier: MonotoneClassifier,
    *,
    shift_factor: float = 3.0,
    mix_original: float = 0.3,
    rng: np.random.Generator,
) -> torch.Tensor:
    """
    Build a sampling distribution that concentrates near the classifier's
    decision boundary (where unknowns are most likely).

    For components the classifier deems important for the survival/failure
    split, shifts probability mass toward degraded states (increasing the
    chance of sampling near the boundary from the survival side).

    Args:
        probs: (n_vars, n_state) original component probabilities (torch)
        classifier: trained MonotoneClassifier
        shift_factor: aggressiveness of the shift toward degraded states
        mix_original: fraction of original distribution to mix in (exploration)
        rng: numpy random generator

    Returns:
        boundary_probs: (n_vars, n_state) shifted probability tensor
    """
    n_vars, n_state = probs.shape
    orig = probs.cpu().numpy().astype(np.float64)

    # Extract feature importance from classifier
    importance = np.zeros(n_vars)
    if classifier._fitted and hasattr(classifier._clf, '_predictors'):
        for predictors_at_iter in classifier._clf._predictors:
            for predictor in predictors_at_iter:
                nodes = predictor.nodes
                if hasattr(nodes, 'dtype') and 'feature_idx' in nodes.dtype.names:
                    for node in nodes:
                        is_leaf = (bool(node['is_leaf'])
                                   if 'is_leaf' in nodes.dtype.names else False)
                        if not is_leaf:
                            fi = int(node['feature_idx'])
                            if 0 <= fi < n_vars:
                                importance[fi] += 1
        total = importance.sum()
        if total > 0:
            importance /= total

    shifted = np.empty_like(orig)
    for i in range(n_vars):
        row = orig[i].copy()
        row_sum = row.sum()
        if row_sum > 0:
            row /= row_sum

        # When importance is all zeros (no failures seen / unfitted),
        # apply uniform shift so sampling still explores degraded states.
        if importance.max() > 0:
            comp_shift = shift_factor * (importance[i] / importance.max())
        else:
            comp_shift = shift_factor

        # Shift mass toward lower (degraded) states
        n_active = int((row > 0).sum())
        max_s = max(n_active - 1, 0)
        weights = np.array([np.exp(comp_shift * (max_s - s))
                            for s in range(n_state)])
        row_shifted = row * weights
        row_sum = row_shifted.sum()
        if row_sum > 0:
            row_shifted /= row_sum
        else:
            row_shifted = row.copy()

        # Mix with original for exploration
        shifted[i] = (1.0 - mix_original) * row_shifted + mix_original * row

        # Ensure no zero probabilities where original is nonzero
        shifted[i] = np.clip(shifted[i], 1e-10, None)
        shifted[i] /= shifted[i].sum()

    return torch.tensor(shifted, dtype=probs.dtype, device=probs.device)


def sample_boundary_candidates(
    probs: torch.Tensor,
    classifier: MonotoneClassifier,
    n_samples: int,
    *,
    shift_factor: float = 3.0,
    mix_original: float = 0.3,
    rng: np.random.Generator,
    boundary_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate one-hot encoded samples biased toward the classifier's decision
    boundary.  Drop-in replacement for ``sample_categorical`` in TSUM's
    search phase.

    Args:
        probs: (n_vars, n_state) original component probabilities
        classifier: trained MonotoneClassifier
        n_samples: number of samples to generate
        shift_factor: IS shift aggressiveness
        mix_original: fraction of original distribution to preserve
        rng: numpy random generator
        boundary_probs: pre-computed shifted distribution (avoids rebuilding)

    Returns:
        samples: (n_samples, n_vars, n_state) one-hot encoded, same device as probs
    """
    from tsum.tsum import sample_categorical

    if boundary_probs is None:
        boundary_probs = build_boundary_distribution(
            probs, classifier,
            shift_factor=shift_factor, mix_original=mix_original, rng=rng)

    return sample_categorical(boundary_probs, n_samples)


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

    Returns:
        X_selected: (n_select, n_vars) component-state vectors to evaluate
    """
    X_cand = sample_component_states(
        probs_dict, row_names, n_candidates, n_state, rng)

    p_fail = classifier.predict_proba_failure(X_cand)

    # Uncertainty = closeness to 0.5
    uncertainty = 1.0 - 2.0 * np.abs(p_fail - 0.5)

    # Also weight by failure probability to explore failure region
    score = uncertainty + 0.5 * p_fail

    n_select = min(n_select, len(X_cand))
    top_idx = np.argsort(score)[-n_select:]
    return X_cand[top_idx]


# ── Integration: BoundaryGuide for use inside run_rule_extraction_by_mcs ────

class BoundaryGuide:
    """
    Manages classifier training and boundary-guided sampling within the
    TSUM rule extraction loop.

    Usage inside ``run_rule_extraction_by_mcs``::

        guide = BoundaryGuide(n_vars, n_state, probs, row_names, sfun)

        # Phase 0 (optional): pre-train on initial random sfun evaluations
        guide.pretrain(n_samples=5000, n_workers=4)

        # In the search loop, replace sample_categorical:
        samples = guide.generate_candidates(batch_size)

        # After minimization produces (comps_st, sys_st), feed back:
        guide.add_observation(comps_st, sys_st)

        # Periodically retrain (e.g. every N rounds):
        guide.retrain()
    """

    def __init__(
        self,
        n_vars: int,
        n_state: int,
        probs: torch.Tensor,
        row_names: List[str],
        sfun: Callable,
        *,
        shift_factor: float = 3.0,
        mix_original: float = 0.3,
        seed: int = 42,
    ):
        self.n_vars = n_vars
        self.n_state = n_state
        # Ensure probs are on the same device TSUM uses for rules
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.probs = probs.to(self._device)
        self.row_names = row_names
        self.sfun = sfun
        self.shift_factor = shift_factor
        self.mix_original = mix_original
        self.rng = np.random.default_rng(seed)

        self.classifier = MonotoneClassifier(n_vars, n_state)
        self._X_data: List[np.ndarray] = []
        self._y_data: List[int] = []
        self._boundary_probs: Optional[torch.Tensor] = None
        self._needs_retrain = False

    @property
    def n_observations(self) -> int:
        return len(self._y_data)

    @property
    def n_failures(self) -> int:
        return sum(1 for y in self._y_data if y == 0)

    @property
    def fitted(self) -> bool:
        return self.classifier._fitted

    def add_observation(self, comps_st: Dict[str, int], sys_st: int):
        """Record an sfun evaluation result for classifier training."""
        x = np.array([comps_st.get(n, 0) for n in self.row_names], dtype=np.int32)
        self._X_data.append(x)
        self._y_data.append(int(sys_st))
        self._needs_retrain = True

    def add_observations_batch(self, X: np.ndarray, y: np.ndarray):
        """Record a batch of sfun evaluations."""
        for i in range(len(y)):
            self._X_data.append(X[i])
            self._y_data.append(int(y[i]))
        self._needs_retrain = True

    def pretrain(self, n_samples: int = 5000, n_workers: int = 1):
        """
        Generate initial training data by random sampling + sfun evaluation,
        then train the classifier.
        """
        probs_dict = {}
        for i, name in enumerate(self.row_names):
            p_row = self.probs[i].cpu().numpy()
            probs_dict[name] = {str(s): {"p": float(p_row[s])}
                                for s in range(self.n_state) if p_row[s] > 0}

        X = sample_component_states(
            probs_dict, self.row_names, n_samples, self.n_state, self.rng)
        y = evaluate_sfun_batch(X, self.sfun, self.row_names, n_workers)
        self.add_observations_batch(X, y)
        self.retrain()

        n_fail = int((y == 0).sum())
        print(f"  BoundaryGuide pretrained: {n_samples} samples, "
              f"{n_fail} failures, accuracy={self.classifier.score(X, y):.4f}")

    def retrain(self):
        """Retrain classifier on all accumulated observations."""
        if len(self._y_data) < 4:
            return
        X = np.array(self._X_data, dtype=np.int32)
        y = np.array(self._y_data, dtype=np.int32)

        # Need both classes with enough members for stratified split
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2 or counts.min() < 2:
            self._needs_retrain = False
            return

        self.classifier.fit(X, y)
        self._boundary_probs = None  # invalidate cached distribution
        self._needs_retrain = False

    def generate_candidates(self, n_samples: int) -> torch.Tensor:
        """
        Generate one-hot samples biased toward the decision boundary.

        Falls back to ``sample_categorical(probs)`` when classifier is unfitted.

        Returns:
            samples: (n_samples, n_vars, n_state) one-hot tensor
        """
        from tsum.tsum import sample_categorical

        if not self.classifier._fitted:
            return sample_categorical(self.probs, n_samples)

        if self._boundary_probs is None:
            self._boundary_probs = build_boundary_distribution(
                self.probs, self.classifier,
                shift_factor=self.shift_factor,
                mix_original=self.mix_original,
                rng=self.rng)

        return sample_categorical(self._boundary_probs, n_samples)
