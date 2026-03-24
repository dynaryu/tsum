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
# Topology analysis
# ---------------------------------------------------------------------------

def analyze_topology(graph, n_edges: int) -> Dict:
    """
    Analyze graph topology to predict whether ML guidance will help.

    ML-guided optimization works best when rules are sparse — i.e., when
    only a few components are critical to system survival/failure. This
    correlates with topological properties like low edge connectivity
    and low min-cut to total-edges ratio.

    Args:
        graph: networkx.Graph object
        n_edges: total number of edges

    Returns:
        dict with:
            - edge_connectivity: min number of edges whose removal disconnects the graph
            - min_cut_ratio: edge_connectivity / n_edges
            - predicted_rule_density: estimated fraction of components per rule
            - ml_recommended: bool — whether ML guidance is likely to help
            - reason: str — explanation of the recommendation
    """
    import networkx as nx

    result = {
        "edge_connectivity": None,
        "min_cut_ratio": None,
        "predicted_rule_density": None,
        "ml_recommended": False,
        "reason": "",
    }

    try:
        ec = nx.edge_connectivity(graph)
    except nx.NetworkXError:
        result["reason"] = "Could not compute edge connectivity"
        return result

    result["edge_connectivity"] = ec
    result["min_cut_ratio"] = ec / n_edges if n_edges > 0 else 0.0

    # Heuristic: predicted rule density correlates with how redundant the graph is.
    # Low edge connectivity → few critical edges → sparse rules.
    # High edge connectivity → many redundant paths → dense rules.
    #
    # For connectivity problems, a survival rule is roughly a spanning path/tree,
    # and a failure rule is roughly a cut set. The min-cut size gives the smallest
    # failure rule. Survival rules tend to be denser — they need enough edges to
    # form a connected subgraph.
    #
    # Empirical observations:
    #   - rg1 (263 edges, ec=2): avg_rule_len=59/263=0.22 — dense, ML unhelpful
    #   - series-parallel (13 edges, ec=1): avg_rule_len=3/13=0.23 — but small, so fast anyway
    #
    # The key signal is: if min_cut_ratio is very small, the graph has bottleneck
    # edges that dominate failure rules, and ML can learn which edges matter.
    # If min_cut_ratio is larger, the graph is uniformly redundant.

    min_cut_ratio = result["min_cut_ratio"]

    # Estimate rule density from graph properties.
    # For connectivity problems:
    #   - Failure rules ≈ cut sets (size ≈ edge_connectivity)
    #   - Survival rules ≈ paths/spanning structures
    #
    # The key insight from rg1 (263 edges, ec=2, avg_rule_len=59/263=0.22):
    # Even with low edge connectivity, survival rules can be DENSE because
    # the greedy minimiser finds long paths through redundant regions.
    #
    # Better predictor: average node degree relative to diameter.
    # High avg_degree + short diameter → many redundant paths → dense survival rules.
    # Low avg_degree + long diameter → few paths → sparse survival rules.

    n_nodes = graph.number_of_nodes() if graph.number_of_nodes() > 0 else 1
    avg_degree = 2 * n_edges / n_nodes

    try:
        diameter = nx.diameter(graph)
    except nx.NetworkXError:
        diameter = n_nodes - 1

    # Predicted survival rule density: path_length / n_edges
    # A survival rule is roughly a path from source to sink.
    # path_length ≈ diameter (worst case), but in dense graphs with short diameter,
    # the greedy minimiser often can't reduce below a large set.
    # Better estimate: diameter * avg_degree / n_edges captures both path length
    # and local redundancy.
    if n_edges > 0:
        predicted_density = min(diameter * avg_degree / n_edges, 1.0)
    else:
        predicted_density = 1.0
    result["predicted_rule_density"] = predicted_density

    if n_edges < 20:
        result["reason"] = f"Graph too small ({n_edges} edges) — ML overhead not worthwhile"
        return result

    # Decision logic:
    # ML helps when rules are sparse, meaning the graph has clear bottlenecks
    # and low redundancy. We combine multiple signals.

    if ec <= 2 and avg_degree < 6:
        # Low connectivity AND low degree: clear bottleneck structure
        result["ml_recommended"] = True
        result["reason"] = (
            f"Low edge connectivity ({ec}) with low avg degree ({avg_degree:.1f}). "
            f"Bottleneck edges exist — ML can identify critical components."
        )
    elif predicted_density < 0.15:
        # Sparse rules predicted from topology
        result["ml_recommended"] = True
        result["reason"] = (
            f"Predicted sparse rules (density={predicted_density:.3f}). "
            f"ML-guided component ordering should help."
        )
    else:
        result["ml_recommended"] = False
        result["reason"] = (
            f"High redundancy (edge_conn={ec}, avg_degree={avg_degree:.1f}, "
            f"pred_density={predicted_density:.3f}). Rules likely dense — "
            f"random sampling is more efficient than ML guidance."
        )

    return result


# ---------------------------------------------------------------------------
# Auto-determination
# ---------------------------------------------------------------------------

# Runtime thresholds (checked during execution when topology analysis is not available)
_MIN_PROBLEM_SIZE = 40      # n_edges * n_state
_MIN_RULES_FOR_ML = 10      # need enough training signal
_MIN_ROUNDS_FOR_ML = 20     # let random sampling run first to establish baseline


def should_use_ml_guidance(
    n_edges: int,
    n_state: int,
    n_rules: int,
    *,
    n_rounds: int = 0,
    avg_rule_len: float = 0.0,
    topology_recommendation: Optional[bool] = None,
    override: Optional[bool] = None,
) -> bool:
    """
    Decide whether ML-guided sampling/minimisation should be active.

    Uses topology_recommendation (from analyze_topology) as the primary signal
    when available. Falls back to runtime heuristics (rule density, round count)
    otherwise.

    Args:
        n_edges: number of components/edges
        n_state: number of states per component
        n_rules: current total number of survival + failure rules
        n_rounds: current round number
        avg_rule_len: average number of conditions per rule
        topology_recommendation: result of analyze_topology()["ml_recommended"]
        override: True = always on, False = always off, None = auto
    """
    if not _HAS_SKLEARN:
        return False
    if override is not None:
        return override

    # If topology analysis says no, respect it
    if topology_recommendation is False:
        return False

    # Basic size check
    if n_edges * n_state < _MIN_PROBLEM_SIZE:
        return False

    # If topology says yes, still need minimum rules and rounds
    if n_rules < _MIN_RULES_FOR_ML:
        return False
    if n_rounds < _MIN_ROUNDS_FOR_ML:
        return False

    # If topology analysis was done and recommended ML, trust it
    if topology_recommendation is True:
        return True

    # No topology info: fall back to runtime rule density check
    if avg_rule_len > 0 and avg_rule_len >= n_edges * 0.25:
        return False

    return True


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
