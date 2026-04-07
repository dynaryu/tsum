"""Tests for tsum.classifier module."""

import numpy as np
import torch
import pytest

from tsum.classifier import (
    MonotoneClassifier,
    sample_component_states,
    compute_is_weights,
    build_boundary_distribution,
    sample_boundary_candidates,
    select_active_samples,
    evaluate_sfun_batch,
    BoundaryGuide,
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


def _make_probs_tensor(probs_dict, n_state):
    """Convert probs_dict to (n_vars, n_state) torch tensor."""
    row_names = list(probs_dict.keys())
    rows = []
    for name in row_names:
        p = probs_dict[name]
        rows.append([p[str(s)]["p"] if str(s) in p else 0.0
                     for s in range(n_state)])
    return torch.tensor(rows, dtype=torch.float32)


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
        probs_dict = {"x0": {"0": {"p": 0.3}, "1": {"p": 0.7}}}
        row_names = ["x0"]
        is_probs_list = [[0.5, 0.5]]

        rng = np.random.default_rng(42)
        # Sample from IS distribution using sample_component_states with IS probs
        is_dict = {"x0": {"0": {"p": 0.5}, "1": {"p": 0.5}}}
        X = sample_component_states(is_dict, row_names, 50000, 2, rng)
        w = compute_is_weights(X, probs_dict, is_probs_list, row_names, 2)

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


# ── Boundary distribution ────────────────────────────────────────────────────

class TestBoundaryDistribution:

    def test_boundary_probs_valid(self):
        probs_dict = _make_probs_dict(n_vars=3, n_state=2)
        probs_t = _make_probs_tensor(probs_dict, 2)

        clf = MonotoneClassifier(3, 2)
        X = np.array([[0, 0, 0], [1, 1, 1]])
        y = np.array([0, 1])
        clf.fit(X, y)

        rng = np.random.default_rng(0)
        bp = build_boundary_distribution(probs_t, clf, rng=rng)
        assert bp.shape == (3, 2)
        for i in range(3):
            assert abs(bp[i].sum().item() - 1.0) < 1e-6
            assert torch.all(bp[i] > 0)

    def test_shift_increases_failure_probability(self):
        """Boundary distribution should assign more mass to degraded states."""
        probs_dict = _make_probs_dict(n_vars=3, n_state=2)
        probs_t = _make_probs_tensor(probs_dict, 2)

        clf = MonotoneClassifier(3, 2)
        rng = np.random.default_rng(42)
        X = rng.integers(0, 2, size=(200, 3))
        y = (X[:, 0] >= 1).astype(int)  # only x0 matters
        clf.fit(X, y)

        bp = build_boundary_distribution(
            probs_t, clf, shift_factor=5.0, rng=rng)
        # x0 should have higher P(state=0) than original 0.1
        assert bp[0, 0].item() > 0.1, \
            f"Boundary P(x0=0)={bp[0, 0].item()} should be > 0.1"

    def test_sample_boundary_candidates_shape(self):
        probs_dict = _make_probs_dict(n_vars=4, n_state=2)
        probs_t = _make_probs_tensor(probs_dict, 2)

        clf = MonotoneClassifier(4, 2)
        X = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        y = np.array([0, 1])
        clf.fit(X, y)

        rng = np.random.default_rng(0)
        samples = sample_boundary_candidates(probs_t, clf, 100, rng=rng)
        assert samples.shape == (100, 4, 2)
        # Each sample should be one-hot per variable
        assert torch.all(samples.sum(dim=2) == 1)


# ── BoundaryGuide ────────────────────────────────────────────────────────────

class TestBoundaryGuide:

    def test_pretrain_and_generate(self):
        n_vars = 4
        n_state = 2
        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        probs_t = _make_probs_tensor(probs_dict, n_state)
        row_names = list(probs_dict.keys())
        sfun = _make_sum_sfun(threshold=3)

        guide = BoundaryGuide(n_vars, n_state, probs_t, row_names, sfun)
        guide.pretrain(n_samples=500)

        assert guide.fitted
        assert guide.n_observations == 500

        samples = guide.generate_candidates(100)
        assert samples.shape == (100, n_vars, n_state)
        assert torch.all(samples.sum(dim=2) == 1)

    def test_add_observation_and_retrain(self):
        n_vars = 3
        n_state = 2
        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        probs_t = _make_probs_tensor(probs_dict, n_state)
        row_names = list(probs_dict.keys())
        sfun = _make_sum_sfun(threshold=2)

        guide = BoundaryGuide(n_vars, n_state, probs_t, row_names, sfun)

        # Add some observations manually
        guide.add_observation({"x0": 0, "x1": 0, "x2": 0}, 0)
        guide.add_observation({"x0": 1, "x1": 1, "x2": 1}, 1)
        guide.add_observation({"x0": 0, "x1": 1, "x2": 1}, 1)
        guide.add_observation({"x0": 1, "x1": 0, "x2": 0}, 0)

        assert guide.n_observations == 4
        assert guide.n_failures == 2

        guide.retrain()
        assert guide.fitted

    def test_seed_from_failure_rules(self):
        """Seeding with failure rules should add failure observations."""
        n_vars = 4
        n_state = 2
        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        probs_t = _make_probs_tensor(probs_dict, n_state)
        row_names = list(probs_dict.keys())
        sfun = _make_sum_sfun(threshold=3)

        guide = BoundaryGuide(n_vars, n_state, probs_t, row_names, sfun)

        # Seed with failure rules in the format from k-fixed search
        seed_rules = [
            {"x0": ["<=", 0], "x1": ["<=", 0], "sys": ["<=", 0]},
            {"x2": ["<=", 0], "x3": ["<=", 0], "sys": ["<=", 0]},
        ]
        guide.seed_from_failure_rules(seed_rules)

        assert guide.n_observations == 2
        assert guide.n_failures == 2

        # The seeded observations should have correct values
        # Rule 1: x0=0, x1=0, x2=1(max), x3=1(max)
        np.testing.assert_array_equal(guide._X_data[0], [0, 0, 1, 1])
        # Rule 2: x0=1(max), x1=1(max), x2=0, x3=0
        np.testing.assert_array_equal(guide._X_data[1], [1, 1, 0, 0])

    def test_seed_then_pretrain_enables_fitting(self):
        """Seeding + pretrain should give enough failure signal to fit."""
        n_vars = 4
        n_state = 2
        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        probs_t = _make_probs_tensor(probs_dict, n_state)
        row_names = list(probs_dict.keys())
        sfun = _make_sum_sfun(threshold=1)  # very rare failure

        guide = BoundaryGuide(n_vars, n_state, probs_t, row_names, sfun)

        # Seed with failure rules to ensure at least 2 failure observations
        seed_rules = [
            {"x0": ["<=", 0], "x1": ["<=", 0], "sys": ["<=", 0]},
            {"x2": ["<=", 0], "x3": ["<=", 0], "sys": ["<=", 0]},
        ]
        guide.seed_from_failure_rules(seed_rules)
        guide.pretrain(n_samples=100)

        # With seeds, classifier should be fitted even if random pretrain
        # finds no failures
        assert guide.fitted
        assert guide.n_failures >= 2

    def test_rank_unknowns_prioritises_failures(self):
        """rank_unknowns should pick states closest to failure boundary."""
        n_vars = 4
        n_state = 2
        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        probs_t = _make_probs_tensor(probs_dict, n_state)
        row_names = list(probs_dict.keys())
        # threshold=3 means sum<3 is failure — ~34% failure rate with P(ok)=0.9
        sfun = _make_sum_sfun(threshold=3)

        guide = BoundaryGuide(n_vars, n_state, probs_t, row_names, sfun)
        guide.pretrain(n_samples=500)

        # Create a batch of one-hot samples: mix of mostly-operational and mostly-failed
        samples = torch.zeros(10, n_vars, n_state)
        for i in range(10):
            for j in range(n_vars):
                # First 5 samples: all operational (state=1)
                # Last 5 samples: all failed (state=0) — these should be ranked higher
                s = 1 if i < 5 else 0
                samples[i, j, s] = 1

        idx_unknown = torch.arange(10)
        picked = guide.rank_unknowns(samples, idx_unknown, n_pick=3)

        assert len(picked) == 3
        # The picked indices should prefer the failed states (indices 5-9)
        # since they have higher P(failure)
        for idx in picked:
            assert idx.item() >= 5, \
                f"Expected failed-state index (>=5), got {idx.item()}"

    def test_unfitted_falls_back_to_original(self):
        """Before training, generate_candidates should use original distribution."""
        n_vars = 3
        n_state = 2
        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        probs_t = _make_probs_tensor(probs_dict, n_state)
        row_names = list(probs_dict.keys())
        sfun = _make_sum_sfun(threshold=2)

        guide = BoundaryGuide(n_vars, n_state, probs_t, row_names, sfun)
        assert not guide.fitted

        # Should not raise, falls back to sample_categorical
        samples = guide.generate_candidates(50)
        assert samples.shape == (50, n_vars, n_state)


# ── End-to-end with run_rule_extraction_by_mcs ──────────────────────────────

class TestEndToEnd:

    def test_classifier_guided_extraction(self):
        """Test that classifier_guided=True works in run_rule_extraction_by_mcs."""
        from tsum.tsum import run_rule_extraction_by_mcs

        n_vars = 4
        n_state = 2
        threshold = 3  # failure when sum < 3

        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        row_names = list(probs_dict.keys())
        probs_t = _make_probs_tensor(probs_dict, n_state)
        sfun = _make_sum_sfun(threshold=threshold)

        result = run_rule_extraction_by_mcs(
            sfun=sfun,
            probs=probs_t,
            row_names=row_names,
            n_state=n_state,
            sys_surv_st=1,
            unk_prob_thres=1e-1,
            unk_prob_opt="abs",
            n_sample=100_000,
            sample_batch_size=50_000,
            classifier_guided=True,
            classifier_n_pretrain=500,
            classifier_retrain_every=5,
            output_dir="/tmp/test_classifier_guided",
        )

        # Should have completed with metrics
        assert len(result["metrics_log"]) > 0
        # Should have found rules (saved to disk)
        import json
        surv_rules = json.loads(open(result["rules_surv_path"]).read())
        fail_rules = json.loads(open(result["rules_fail_path"]).read())
        assert len(surv_rules) > 0 or len(fail_rules) > 0

    def test_degradation_ranked_extraction(self):
        """Test that rank_by_degradation=True works in run_rule_extraction_by_mcs."""
        from tsum.tsum import run_rule_extraction_by_mcs

        n_vars = 4
        n_state = 2
        threshold = 3  # failure when sum < 3

        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        row_names = list(probs_dict.keys())
        probs_t = _make_probs_tensor(probs_dict, n_state)
        sfun = _make_sum_sfun(threshold=threshold)

        result = run_rule_extraction_by_mcs(
            sfun=sfun,
            probs=probs_t,
            row_names=row_names,
            n_state=n_state,
            sys_surv_st=1,
            unk_prob_thres=1e-1,
            unk_prob_opt="abs",
            n_sample=100_000,
            sample_batch_size=50_000,
            rank_by_degradation=True,
            output_dir="/tmp/test_degradation_ranked",
        )

        assert len(result["metrics_log"]) > 0
        import json
        surv_rules = json.loads(open(result["rules_surv_path"]).read())
        fail_rules = json.loads(open(result["rules_fail_path"]).read())
        assert len(surv_rules) > 0 or len(fail_rules) > 0

    def test_sensitivity_prescreen(self):
        """Test that sensitivity_prescreen runs and initialises weights."""
        from tsum.tsum import run_rule_extraction_by_mcs

        n_vars = 4
        n_state = 2
        threshold = 3

        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        row_names = list(probs_dict.keys())
        probs_t = _make_probs_tensor(probs_dict, n_state)
        sfun = _make_sum_sfun(threshold=threshold)

        result = run_rule_extraction_by_mcs(
            sfun=sfun,
            probs=probs_t,
            row_names=row_names,
            n_state=n_state,
            sys_surv_st=1,
            unk_prob_thres=1e-1,
            unk_prob_opt="abs",
            n_sample=100_000,
            sample_batch_size=50_000,
            rank_by_degradation=True,
            sensitivity_prescreen=True,
            output_dir="/tmp/test_sensitivity_prescreen",
        )

        assert len(result["metrics_log"]) > 0

    def test_degradation_diversity(self):
        """Test that diversity selection (probabilistic) works without error."""
        from tsum.tsum import run_rule_extraction_by_mcs

        n_vars = 4
        n_state = 2
        threshold = 3

        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        row_names = list(probs_dict.keys())
        probs_t = _make_probs_tensor(probs_dict, n_state)
        sfun = _make_sum_sfun(threshold=threshold)

        result = run_rule_extraction_by_mcs(
            sfun=sfun,
            probs=probs_t,
            row_names=row_names,
            n_state=n_state,
            sys_surv_st=1,
            unk_prob_thres=1e-1,
            unk_prob_opt="abs",
            n_sample=100_000,
            sample_batch_size=50_000,
            rank_by_degradation=True,
            degradation_diversity=True,
            output_dir="/tmp/test_degradation_diversity",
        )

        assert len(result["metrics_log"]) > 0

    def test_is_sampling_with_seeds(self):
        """Test that is_sampling=True with seed rules biases the search phase."""
        from tsum.tsum import run_rule_extraction_by_mcs

        n_vars = 4
        n_state = 2
        threshold = 3

        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        row_names = list(probs_dict.keys())
        probs_t = _make_probs_tensor(probs_dict, n_state)
        sfun = _make_sum_sfun(threshold=threshold)

        # Provide a seed rule so component importance is nonzero
        seed_rules = [{"x0": ["<=", 0], "x1": ["<=", 0], "sys": ["<=", 0]}]

        result = run_rule_extraction_by_mcs(
            sfun=sfun,
            probs=probs_t,
            row_names=row_names,
            n_state=n_state,
            sys_surv_st=1,
            unk_prob_thres=1e-1,
            unk_prob_opt="abs",
            n_sample=100_000,
            sample_batch_size=50_000,
            rank_by_degradation=True,
            classifier_seed_rules=seed_rules,
            is_sampling=True,
            is_shift_factor=3.0,
            is_mix_original=0.3,
            is_rebuild_every=5,
            output_dir="/tmp/test_is_sampling_seeds",
        )

        assert len(result["metrics_log"]) > 0

    def test_is_sampling_skipped_without_signal(self):
        """is_sampling without any importance signal should run safely (skipped)."""
        from tsum.tsum import run_rule_extraction_by_mcs

        n_vars = 4
        n_state = 2
        threshold = 3

        probs_dict = _make_probs_dict(n_vars=n_vars, n_state=n_state)
        row_names = list(probs_dict.keys())
        probs_t = _make_probs_tensor(probs_dict, n_state)
        sfun = _make_sum_sfun(threshold=threshold)

        result = run_rule_extraction_by_mcs(
            sfun=sfun,
            probs=probs_t,
            row_names=row_names,
            n_state=n_state,
            sys_surv_st=1,
            unk_prob_thres=1e-1,
            unk_prob_opt="abs",
            n_sample=100_000,
            sample_batch_size=50_000,
            rank_by_degradation=True,
            is_sampling=True,
            output_dir="/tmp/test_is_sampling_no_signal",
        )

        assert len(result["metrics_log"]) > 0
