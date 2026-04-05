"""Tests for tsum.classifier module."""

import numpy as np
import pytest

from tsum.classifier import (
    MonotoneClassifier,
    sample_component_states,
    compute_is_weights,
    build_is_distribution,
    sample_is,
    select_active_samples,
    evaluate_sfun_batch,
    estimate_failure_probability,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_probs_dict(n_vars=4, n_state=2):
    """Simple binary components with P(fail)=0.1, P(ok)=0.9."""
    probs = {}
    for i in range(n_vars):
        probs[f"x{i}"] = {
            "0": {"p": 0.1},
            "1": {"p": 0.9},
        }
    return probs


def _make_probs_dict_multistate(n_vars=4, n_state=3):
    """Multi-state components."""
    probs = {}
    for i in range(n_vars):
        d = {}
        for s in range(n_state):
            d[str(s)] = {"p": 1.0 / n_state}
        probs[f"x{i}"] = d
    return probs


def _sum_sfun(comps_st, threshold=2):
    """
    Simple coherent sfun: system survives (sys_st=1) if sum of states >= threshold.
    """
    total = sum(v for k, v in comps_st.items())
    sys_st = 1 if total >= threshold else 0
    return float(total), sys_st, None


def _make_sum_sfun(threshold):
    def sfun(comps_st):
        return _sum_sfun(comps_st, threshold=threshold)
    return sfun


# ── MonotoneClassifier ────────────────────────────────────────────────────────

class TestMonotoneClassifier:

    def test_fit_predict_basic(self):
        clf = MonotoneClassifier(n_vars=3, n_state=2)
        # Simple data: system fails when all components are 0
        X = np.array([[0, 0, 0], [1, 1, 1], [0, 1, 1], [1, 0, 1]])
        y = np.array([0, 1, 1, 1])
        clf.fit(X, y)
        assert clf._fitted

        pred = clf.predict(X)
        assert pred[0] == 0  # all-zero should be failure
        assert pred[1] == 1  # all-one should be survival

    def test_monotonicity(self):
        """Better component states should never increase P(failure)."""
        clf = MonotoneClassifier(n_vars=4, n_state=3)
        rng = np.random.default_rng(42)
        X = rng.integers(0, 3, size=(500, 4))
        y = (X.sum(axis=1) >= 5).astype(int)
        clf.fit(X, y)

        # Compare x_low <= x_high componentwise
        x_low = np.array([[0, 0, 1, 0]])
        x_high = np.array([[1, 1, 2, 1]])

        p_fail_low = clf.predict_proba_failure(x_low)[0]
        p_fail_high = clf.predict_proba_failure(x_high)[0]
        assert p_fail_low >= p_fail_high, \
            f"Monotonicity violated: P(fail|low)={p_fail_low} < P(fail|high)={p_fail_high}"

    def test_predict_proba_range(self):
        clf = MonotoneClassifier(n_vars=2, n_state=2)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        clf.fit(X, y)

        p = clf.predict_proba_failure(X)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_unfitted_returns_defaults(self):
        clf = MonotoneClassifier(n_vars=2, n_state=2)
        X = np.array([[0, 1], [1, 0]])
        p = clf.predict_proba_failure(X)
        assert np.allclose(p, 0.5)
        pred = clf.predict(X)
        assert np.all(pred == 1)


# ── Sampling ──────────────────────────────────────────────────────────────────

class TestSampling:

    def test_sample_shape(self):
        probs = _make_probs_dict(n_vars=5, n_state=2)
        rng = np.random.default_rng(0)
        X = sample_component_states(probs, list(probs.keys()), 100, 2, rng)
        assert X.shape == (100, 5)
        assert X.dtype == np.int32

    def test_sample_values_in_range(self):
        probs = _make_probs_dict_multistate(n_vars=3, n_state=3)
        rng = np.random.default_rng(0)
        X = sample_component_states(probs, list(probs.keys()), 1000, 3, rng)
        assert np.all(X >= 0)
        assert np.all(X < 3)

    def test_sample_distribution(self):
        """Check that sampling respects the probability distribution."""
        probs = {"x0": {"0": {"p": 0.3}, "1": {"p": 0.7}}}
        rng = np.random.default_rng(123)
        X = sample_component_states(probs, ["x0"], 50000, 2, rng)
        frac_zero = (X[:, 0] == 0).mean()
        assert abs(frac_zero - 0.3) < 0.02, f"Expected ~0.3, got {frac_zero}"


# ── IS weights ────────────────────────────────────────────────────────────────

class TestISWeights:

    def test_weights_equal_one_when_same_distribution(self):
        probs = _make_probs_dict(n_vars=3, n_state=2)
        row_names = list(probs.keys())
        is_probs = [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]

        rng = np.random.default_rng(0)
        X = sample_component_states(probs, row_names, 100, 2, rng)
        w = compute_is_weights(X, probs, is_probs, row_names, 2)
        np.testing.assert_allclose(w, 1.0, atol=1e-10)

    def test_weights_positive(self):
        probs = _make_probs_dict(n_vars=3, n_state=2)
        row_names = list(probs.keys())
        is_probs = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

        rng = np.random.default_rng(0)
        X = sample_component_states(probs, row_names, 100, 2, rng)
        w = compute_is_weights(X, probs, is_probs, row_names, 2)
        assert np.all(w > 0)

    def test_unbiased_mean(self):
        """IS should give unbiased estimate of E_p[f(x)]."""
        probs = {"x0": {"0": {"p": 0.3}, "1": {"p": 0.7}}}
        row_names = ["x0"]
        is_probs_list = [[0.5, 0.5]]

        rng = np.random.default_rng(42)
        X = sample_is(is_probs_list, 50000, 2, rng)
        w = compute_is_weights(X, probs, is_probs_list, row_names, 2)

        # E_p[X] = 0.7, estimated via IS
        values = X[:, 0].astype(float)
        is_estimate = (w * values).mean()
        assert abs(is_estimate - 0.7) < 0.02, f"IS estimate {is_estimate} != 0.7"


# ── Active learning ───────────────────────────────────────────────────────────

class TestActiveLearning:

    def test_select_returns_correct_shape(self):
        probs = _make_probs_dict(n_vars=4, n_state=2)
        row_names = list(probs.keys())
        clf = MonotoneClassifier(4, 2)
        X = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        y = np.array([0, 1])
        clf.fit(X, y)

        rng = np.random.default_rng(0)
        selected = select_active_samples(probs, row_names, 2, clf, 1000, 50, rng)
        assert selected.shape == (50, 4)


# ── IS distribution ───────────────────────────────────────────────────────────

class TestISDistribution:

    def test_is_probs_valid(self):
        probs = _make_probs_dict(n_vars=3, n_state=2)
        row_names = list(probs.keys())
        clf = MonotoneClassifier(3, 2)
        X = np.array([[0, 0, 0], [1, 1, 1]])
        y = np.array([0, 1])
        clf.fit(X, y)

        rng = np.random.default_rng(0)
        is_probs = build_is_distribution(probs, row_names, 2, clf, rng=rng)
        for p in is_probs:
            assert abs(sum(p) - 1.0) < 1e-8
            assert all(pp > 0 for pp in p)

    def test_shift_increases_failure_probability(self):
        """IS distribution should assign more mass to degraded states."""
        probs = _make_probs_dict(n_vars=3, n_state=2)
        row_names = list(probs.keys())

        clf = MonotoneClassifier(3, 2)
        # Make a classifier that says x0 is important
        rng = np.random.default_rng(42)
        X = rng.integers(0, 2, size=(200, 3))
        y = (X[:, 0] >= 1).astype(int)  # only x0 matters
        clf.fit(X, y)

        is_probs = build_is_distribution(
            probs, row_names, 2, clf, shift_factor=5.0, rng=rng)
        # x0 should have higher P(state=0) than original 0.1
        assert is_probs[0][0] > 0.1, \
            f"IS P(x0=0)={is_probs[0][0]} should be > 0.1"


# ── End-to-end ────────────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_estimate_simple_system(self):
        """Test full pipeline on a simple sum-threshold system."""
        n_vars = 4
        n_state = 2
        threshold = 4  # failure when sum < 4 => only all-ones survives

        probs = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        row_names = list(probs.keys())
        sfun = _make_sum_sfun(threshold=threshold)

        # P(failure) = 1 - 0.9^4 = 0.3439
        expected = 1.0 - 0.9 ** 4

        result = estimate_failure_probability(
            sfun=sfun,
            probs_dict=probs,
            row_names=row_names,
            n_state=n_state,
            n_initial_samples=2000,
            n_active_rounds=2,
            n_active_samples_per_round=500,
            n_active_candidates=5000,
            n_is_samples=50000,
            is_shift_factor=2.0,
            verbose=False,
        )

        assert abs(result.p_failure - expected) < 0.03, \
            f"P(failure)={result.p_failure:.4f}, expected={expected:.4f}"

    def test_estimate_rare_event(self):
        """Test on a system where failure is rare."""
        n_vars = 6
        n_state = 2
        threshold = 1  # failure only when ALL components are 0

        probs = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        row_names = list(probs.keys())
        sfun = _make_sum_sfun(threshold=threshold)

        # P(failure) = 0.1^6 = 1e-6
        expected = 0.1 ** 6

        result = estimate_failure_probability(
            sfun=sfun,
            probs_dict=probs,
            row_names=row_names,
            n_state=n_state,
            n_initial_samples=5000,
            n_active_rounds=3,
            n_active_samples_per_round=1000,
            n_active_candidates=10000,
            n_is_samples=100000,
            is_shift_factor=5.0,
            verbose=False,
        )

        # For very rare events, just check order of magnitude
        assert result.p_failure > 0, "Should detect some failures"
        if result.p_failure > 0:
            log_ratio = abs(np.log10(result.p_failure) - np.log10(expected))
            assert log_ratio < 2.0, \
                f"P(failure)={result.p_failure:.2e}, expected={expected:.2e} (>2 orders off)"
