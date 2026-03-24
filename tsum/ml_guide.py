"""
ML-guided optimizations for TSUM reference state enumeration.

Two optimizations:
1. Boundary-biased sampling: trains a classifier on labeled samples to
   oversample the "unknown" (boundary) region, finding new reference states
   faster.
2. Guided minimisation order: uses feature importance from the classifier
   to order component removal/restoration in the greedy minimiser, reducing
   sfun calls.

The ML model (HistGradientBoostingClassifier from scikit-learn) runs on CPU,
leaving the GPU free for tensor sampling/classification. scikit-learn is an
optional dependency — if not installed, these features are silently disabled.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Auto-determination
# ---------------------------------------------------------------------------

# Minimum problem size (n_edges * n_state) to consider ML guidance
_MIN_PROBLEM_SIZE = 40
# Minimum number of rules before the classifier has enough training signal
_MIN_RULES_FOR_ML = 5


def should_use_ml_guidance(
    n_edges: int,
    n_state: int,
    n_rules: int,
    *,
    override: Optional[bool] = None,
) -> bool:
    """
    Decide whether ML-guided sampling/minimisation should be active.

    Auto logic: enable when the problem is large enough for ML overhead to
    pay off AND enough rules exist for meaningful training data.

    Args:
        n_edges: number of components/edges
        n_state: number of states per component
        n_rules: current total number of survival + failure rules
        override: True = always on, False = always off, None = auto
    """
    if not _HAS_SKLEARN:
        return False
    if override is not None:
        return override
    return (n_edges * n_state >= _MIN_PROBLEM_SIZE) and (n_rules >= _MIN_RULES_FOR_ML)


# ---------------------------------------------------------------------------
# Boundary classifier
# ---------------------------------------------------------------------------

class BoundaryClassifier:
    """
    Lightweight classifier that learns the survival/failure/unknown boundary
    from labeled samples.  Used to:
      - predict P(unknown) for each sample → bias sampling toward boundary
      - extract per-component importance → guide greedy minimisation order
    """

    def __init__(self, n_vars: int, n_state: int):
        self.n_vars = n_vars
        self.n_state = n_state
        self._X: List[np.ndarray] = []   # accumulated feature arrays
        self._y: List[np.ndarray] = []   # accumulated label arrays
        self._clf = None
        self._fitted = False
        # Max training samples to keep (circular buffer behaviour)
        self._max_samples = 200_000

    # -- data accumulation --------------------------------------------------

    def update_training_data(
        self,
        samples: torch.Tensor,
        mask_survival: torch.Tensor,
        mask_failure: torch.Tensor,
        mask_unknown: torch.Tensor,
        *,
        max_per_call: int = 50_000,
    ) -> None:
        """
        Accumulate labeled samples for training.

        Args:
            samples: (B, n_var, n_state) one-hot tensor
            mask_survival/failure/unknown: (B,) bool tensors
        """
        # Convert one-hot → state indices for compact features
        states = torch.argmax(samples, dim=2).cpu().numpy()  # (B, n_var)

        labels = np.full(len(states), 2, dtype=np.int32)  # 2 = unknown
        labels[mask_survival.cpu().numpy()] = 1  # survival
        labels[mask_failure.cpu().numpy()] = 0   # failure

        # Subsample if too many
        if len(states) > max_per_call:
            idx = np.random.choice(len(states), max_per_call, replace=False)
            states = states[idx]
            labels = labels[idx]

        self._X.append(states)
        self._y.append(labels)

        # Enforce max buffer size
        total = sum(x.shape[0] for x in self._X)
        while total > self._max_samples and len(self._X) > 1:
            total -= self._X[0].shape[0]
            self._X.pop(0)
            self._y.pop(0)

    # -- training -----------------------------------------------------------

    def fit(self) -> bool:
        """
        Train (or retrain) the classifier on accumulated data.

        Returns True if fitting succeeded, False if insufficient data.
        """
        if not self._X:
            return False

        X = np.concatenate(self._X, axis=0)
        y = np.concatenate(self._y, axis=0)

        # Need at least 2 distinct classes
        if len(np.unique(y)) < 2:
            return False

        self._clf = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42,
            warm_start=False,
        )
        self._clf.fit(X, y)
        self._fitted = True
        return True

    # -- predictions --------------------------------------------------------

    def predict_unknown_prob(self, samples: torch.Tensor) -> np.ndarray:
        """
        Return P(unknown) for each sample.

        Args:
            samples: (B, n_var, n_state) one-hot tensor

        Returns:
            (B,) numpy array of P(unknown) values in [0, 1]
        """
        if not self._fitted:
            return np.ones(samples.shape[0], dtype=np.float32)

        X = torch.argmax(samples, dim=2).cpu().numpy()
        proba = self._clf.predict_proba(X)  # (B, n_classes)

        # Find the column index for class 2 (unknown)
        classes = list(self._clf.classes_)
        if 2 in classes:
            unk_idx = classes.index(2)
            return proba[:, unk_idx].astype(np.float32)
        else:
            # No unknowns seen during training — return zeros
            return np.zeros(X.shape[0], dtype=np.float32)

    # -- feature importance -------------------------------------------------

    def get_component_importance(self) -> Optional[np.ndarray]:
        """
        Return per-component feature importance vector, shape (n_vars,).

        Higher values mean the component is more influential in determining
        the system state (survival/failure/unknown boundary).

        Uses a cheap tree-based approach: for each split in the ensemble,
        count how often each feature is used. This is O(n_trees) not
        O(n_vars * n_samples) like permutation importance.
        """
        if not self._fitted:
            return None

        # Count feature usage across all trees in the ensemble
        importance = np.zeros(self.n_vars, dtype=np.float64)
        for predictors_at_iter in self._clf._predictors:
            for predictor in predictors_at_iter:
                nodes = predictor.nodes
                if hasattr(nodes, 'dtype') and 'feature_idx' in nodes.dtype.names:
                    for node in nodes:
                        is_leaf = bool(node['is_leaf']) if 'is_leaf' in nodes.dtype.names else False
                        if not is_leaf:
                            fi = int(node['feature_idx'])
                            if 0 <= fi < self.n_vars:
                                importance[fi] += 1

        # Normalize
        total = importance.sum()
        if total > 0:
            importance /= total

        return importance

    def get_minimisation_order_surv(
        self,
        row_names: Sequence[str],
        candidates: List[str],
    ) -> List[str]:
        """
        Return component order for survival minimisation: try removing
        LEAST important components first (most likely to be removable
        without changing survival status).
        """
        importance = self.get_component_importance()
        if importance is None:
            return candidates

        imp_map = {name: importance[i] for i, name in enumerate(row_names)}
        # Sort candidates by importance ASCENDING (least important first)
        return sorted(candidates, key=lambda c: imp_map.get(c, 0.0))

    def get_minimisation_order_fail(
        self,
        row_names: Sequence[str],
        candidates: List[str],
    ) -> List[str]:
        """
        Return component order for failure minimisation: try restoring
        LEAST important components first (most likely to be restorable
        without changing failure status).
        """
        # Same logic as survival: least important first
        return self.get_minimisation_order_surv(row_names, candidates)


# ---------------------------------------------------------------------------
# Boundary-biased sampling
# ---------------------------------------------------------------------------

def ml_biased_sample(
    probs: torch.Tensor,
    sample_batch_size: int,
    classifier: BoundaryClassifier,
    *,
    enrichment_factor: int = 3,
) -> torch.Tensor:
    """
    Generate boundary-enriched samples.

    1. Draw enrichment_factor * sample_batch_size samples from probs
    2. Score each with classifier.predict_unknown_prob()
    3. Resample sample_batch_size with weights proportional to P(unknown)

    If the classifier predicts no unknowns, falls back to uniform resampling
    (equivalent to standard sampling).

    Args:
        probs: (n_var, n_state) probability tensor
        sample_batch_size: target number of output samples
        classifier: trained BoundaryClassifier
        enrichment_factor: oversample multiplier (higher = better enrichment
                          but more memory/compute)

    Returns:
        (sample_batch_size, n_var, n_state) one-hot tensor on same device as probs
    """
    from tsum.tsum import sample_categorical

    n_oversample = enrichment_factor * sample_batch_size
    candidates = sample_categorical(probs, n_oversample)  # (N, n_var, n_state)

    p_unknown = classifier.predict_unknown_prob(candidates)  # (N,)

    # Add small epsilon so even non-boundary samples have a chance
    weights = p_unknown + 1e-6
    weights /= weights.sum()

    # Weighted resample
    indices = np.random.choice(n_oversample, size=sample_batch_size,
                               replace=True, p=weights)
    return candidates[indices]
