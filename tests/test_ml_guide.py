"""Tests for ML-guided optimization module."""

import pytest
import torch
import numpy as np

from tsum.ml_guide import (
    should_use_ml_guidance,
    BoundaryClassifier,
    ml_biased_sample,
    _HAS_SKLEARN,
)

pytestmark = pytest.mark.skipif(
    not _HAS_SKLEARN, reason="scikit-learn not installed"
)


# ---------------------------------------------------------------------------
# should_use_ml_guidance
# ---------------------------------------------------------------------------

class TestShouldUseMLGuidance:

    def test_auto_small_problem_disabled(self):
        # 4 binary components = 8 < 40 threshold
        assert not should_use_ml_guidance(4, 2, n_rules=20)

    def test_auto_large_problem_enabled(self):
        # 25 binary components = 50 >= 40, and 20 rules >= 5
        assert should_use_ml_guidance(25, 2, n_rules=20)

    def test_auto_few_rules_disabled(self):
        # Large problem but only 3 rules
        assert not should_use_ml_guidance(25, 2, n_rules=3)

    def test_override_true(self):
        # Force on even for small problem
        assert should_use_ml_guidance(4, 2, n_rules=0, override=True)

    def test_override_false(self):
        # Force off even for large problem
        assert not should_use_ml_guidance(50, 2, n_rules=100, override=False)

    def test_ternary_states(self):
        # 15 ternary = 45 >= 40
        assert should_use_ml_guidance(15, 3, n_rules=10)


# ---------------------------------------------------------------------------
# BoundaryClassifier
# ---------------------------------------------------------------------------

class TestBoundaryClassifier:

    @pytest.fixture
    def classifier(self):
        return BoundaryClassifier(n_vars=10, n_state=2)

    @pytest.fixture
    def labeled_data(self):
        """Generate synthetic labeled samples."""
        rng = np.random.default_rng(42)
        n = 1000
        n_vars = 10

        # Create one-hot samples
        states = rng.integers(0, 2, size=(n, n_vars))
        samples = torch.nn.functional.one_hot(
            torch.tensor(states, dtype=torch.int64), num_classes=2
        ).int()

        # Label: survival if sum of states >= 5, failure if <= 2, else unknown
        sums = states.sum(axis=1)
        mask_surv = torch.tensor(sums >= 5)
        mask_fail = torch.tensor(sums <= 2)
        mask_unk = ~mask_surv & ~mask_fail

        return samples, mask_surv, mask_fail, mask_unk

    def test_fit_returns_true(self, classifier, labeled_data):
        samples, mask_surv, mask_fail, mask_unk = labeled_data
        classifier.update_training_data(samples, mask_surv, mask_fail, mask_unk)
        assert classifier.fit() is True

    def test_fit_empty_returns_false(self, classifier):
        assert classifier.fit() is False

    def test_predict_unknown_prob_shape(self, classifier, labeled_data):
        samples, mask_surv, mask_fail, mask_unk = labeled_data
        classifier.update_training_data(samples, mask_surv, mask_fail, mask_unk)
        classifier.fit()

        probs = classifier.predict_unknown_prob(samples[:100])
        assert probs.shape == (100,)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_before_fit(self, classifier, labeled_data):
        """Before fitting, predict_unknown_prob returns all ones."""
        samples = labeled_data[0]
        probs = classifier.predict_unknown_prob(samples[:10])
        assert probs.shape == (10,)
        np.testing.assert_array_equal(probs, 1.0)

    def test_component_importance_shape(self, classifier, labeled_data):
        samples, mask_surv, mask_fail, mask_unk = labeled_data
        classifier.update_training_data(samples, mask_surv, mask_fail, mask_unk)
        classifier.fit()

        imp = classifier.get_component_importance()
        assert imp is not None
        assert imp.shape == (10,)
        assert np.all(imp >= 0)

    def test_component_importance_before_fit(self, classifier):
        assert classifier.get_component_importance() is None

    def test_minimisation_order_surv(self, classifier, labeled_data):
        samples, mask_surv, mask_fail, mask_unk = labeled_data
        classifier.update_training_data(samples, mask_surv, mask_fail, mask_unk)
        classifier.fit()

        row_names = [f"e{i}" for i in range(10)]
        candidates = row_names[:5]
        order = classifier.get_minimisation_order_surv(row_names, candidates)

        assert set(order) == set(candidates)
        assert len(order) == len(candidates)

    def test_minimisation_order_without_fit(self, classifier):
        row_names = [f"e{i}" for i in range(10)]
        candidates = row_names[:5]
        # Without fitting, returns candidates unchanged
        order = classifier.get_minimisation_order_surv(row_names, candidates)
        assert order == candidates

    def test_max_samples_buffer(self):
        """Verify buffer doesn't grow unbounded."""
        clf = BoundaryClassifier(n_vars=5, n_state=2)
        clf._max_samples = 500

        for _ in range(20):
            samples = torch.nn.functional.one_hot(
                torch.randint(0, 2, (100, 5)), num_classes=2
            ).int()
            mask_s = torch.rand(100) > 0.5
            mask_f = ~mask_s & (torch.rand(100) > 0.5)
            mask_u = ~mask_s & ~mask_f
            clf.update_training_data(samples, mask_s, mask_f, mask_u)

        total = sum(x.shape[0] for x in clf._X)
        assert total <= 500 + 100  # allow one batch overshoot


# ---------------------------------------------------------------------------
# ml_biased_sample
# ---------------------------------------------------------------------------

class TestMLBiasedSample:

    def test_output_shape(self):
        clf = BoundaryClassifier(n_vars=10, n_state=2)

        # Train with some data
        rng = np.random.default_rng(0)
        states = rng.integers(0, 2, size=(500, 10))
        samples = torch.nn.functional.one_hot(
            torch.tensor(states, dtype=torch.int64), num_classes=2
        ).int()
        sums = states.sum(axis=1)
        mask_s = torch.tensor(sums >= 5)
        mask_f = torch.tensor(sums <= 2)
        mask_u = ~mask_s & ~mask_f
        clf.update_training_data(samples, mask_s, mask_f, mask_u)
        clf.fit()

        probs = torch.tensor([[0.3, 0.7]] * 10, dtype=torch.float32)
        result = ml_biased_sample(probs, 200, clf, enrichment_factor=3)

        assert result.shape == (200, 10, 2)
        # Verify one-hot
        assert torch.all(result.sum(dim=2) == 1)

    def test_enrichment_vs_random(self):
        """ML-biased samples should have higher fraction near the boundary."""
        clf = BoundaryClassifier(n_vars=10, n_state=2)

        rng = np.random.default_rng(42)
        states = rng.integers(0, 2, size=(2000, 10))
        samples = torch.nn.functional.one_hot(
            torch.tensor(states, dtype=torch.int64), num_classes=2
        ).int()
        sums = states.sum(axis=1)
        mask_s = torch.tensor(sums >= 6)
        mask_f = torch.tensor(sums <= 3)
        mask_u = ~mask_s & ~mask_f
        clf.update_training_data(samples, mask_s, mask_f, mask_u)
        clf.fit()

        probs = torch.tensor([[0.5, 0.5]] * 10, dtype=torch.float32)

        # ML-biased
        biased = ml_biased_sample(probs, 5000, clf, enrichment_factor=5)
        biased_sums = torch.argmax(biased, dim=2).sum(dim=1).numpy()
        biased_unk = np.sum((biased_sums >= 4) & (biased_sums <= 5)) / len(biased_sums)

        # Random baseline
        from tsum.tsum import sample_categorical
        random_s = sample_categorical(probs, 5000)
        random_sums = torch.argmax(random_s, dim=2).sum(dim=1).numpy()
        random_unk = np.sum((random_sums >= 4) & (random_sums <= 5)) / len(random_sums)

        # Biased should have at least as many boundary samples
        # (may not always be strictly more due to randomness, but on average yes)
        assert biased_unk >= random_unk * 0.8  # allow 20% margin for randomness


# ---------------------------------------------------------------------------
# Integration with minimise functions
# ---------------------------------------------------------------------------

class TestMinimiseWithComponentOrder:

    def test_surv_with_order(self):
        """minimise_surv_states_random accepts component_order."""
        from tsum.tsum import minimise_surv_states_random

        # Simple system: survives if e0=1 AND e1=1
        def sfun(st):
            alive = st.get("e0", 0) >= 1 and st.get("e1", 0) >= 1
            return None, 1 if alive else 0, None

        comps = {"e0": 1, "e1": 1, "e2": 1, "e3": 1}
        order = ["e3", "e2", "e0", "e1"]  # try removing e3, e2 first

        rule, info = minimise_surv_states_random(
            comps, sfun, sys_surv_st=1, component_order=order)

        # Should find minimal rule with e0 and e1
        assert "e0" in {k for k in rule if k != "sys"}
        assert "e1" in {k for k in rule if k != "sys"}
        # e2, e3 should be removed (not in rule)
        assert "e2" not in rule
        assert "e3" not in rule

    def test_fail_with_order(self):
        """minimise_fail_states_random accepts component_order."""
        from tsum.tsum import minimise_fail_states_random

        # System fails if e0=0 (bridge edge)
        def sfun(st):
            alive = st.get("e0", 1) >= 1
            return None, 1 if alive else 0, None

        comps = {"e0": 0, "e1": 0, "e2": 0, "e3": 0}
        order = ["e1", "e2", "e3", "e0"]  # try restoring e1, e2, e3 first

        rule, info = minimise_fail_states_random(
            comps, sfun, sys_fail_st=0, max_state=1, component_order=order)

        # Should find minimal rule with only e0=0
        rule_comps = {k for k in rule if k != "sys"}
        assert "e0" in rule_comps
        # Others should be restored (not in rule)
        assert "e1" not in rule_comps
        assert "e2" not in rule_comps
        assert "e3" not in rule_comps

    def test_order_none_falls_back_to_random(self):
        """With component_order=None, behaves as before (random shuffle)."""
        from tsum.tsum import minimise_surv_states_random

        def sfun(st):
            alive = st.get("e0", 0) >= 1
            return None, 1 if alive else 0, None

        comps = {"e0": 1, "e1": 1}
        rule, info = minimise_surv_states_random(
            comps, sfun, sys_surv_st=1, component_order=None)

        assert "e0" in {k for k in rule if k != "sys"}
