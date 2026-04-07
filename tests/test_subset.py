"""
Tests for tsum.subset_sim — Au & Beck Subset Simulation with sticky-prior
component-wise Metropolis-Hastings.
"""
from math import comb

import numpy as np
import pytest
import torch

from tsum import subset_sim as ss


# ---------------------------------------------------------------------------
# Synthetic problems
# ---------------------------------------------------------------------------
def _binomial_tail_problem(n_var=20, p_fail=0.05, fail_threshold=8):
    """
    Each component is binary: state 0 = working (prob 1-p_fail), state 1 = failed.
    System fails when at least `fail_threshold` components are in state 1.

    Higher fval (= number of failed components) means more failed, so
    severity_sign = +1.
    """
    probs = torch.tensor([[1.0 - p_fail, p_fail]] * n_var, dtype=torch.float32)

    def sfun(comps_st):
        n_failed = sum(1 for v in comps_st.values() if v == 1)
        fval = float(n_failed)
        sys_st = 1 if n_failed < fail_threshold else 0
        return fval, sys_st, None

    row_names = [f"c{i}" for i in range(n_var)]
    return probs, sfun, row_names


def _true_binomial_tail_prob(n, p, k):
    """P[Bin(n, p) >= k]."""
    return sum(comb(n, j) * (p ** j) * ((1 - p) ** (n - j)) for j in range(k, n + 1))


# ---------------------------------------------------------------------------
# Unit tests for individual helpers
# ---------------------------------------------------------------------------
def test_sample_prior_states_shape_and_range():
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.5, 0.0]])
    states = ss.sample_prior_states(probs, n=200)
    assert states.shape == (200, 2)
    assert states.dtype == torch.int64
    assert int(states.min()) >= 0
    assert int(states.max()) <= 2
    # Component 1 has zero mass on state 2 -> should never be sampled
    assert (states[:, 1] == 2).sum().item() == 0


def test_sample_prior_states_marginal_frequencies():
    """Empirical marginals should match the prior within MC tolerance."""
    torch.manual_seed(42)
    probs = torch.tensor([[0.9, 0.1], [0.3, 0.7]])
    states = ss.sample_prior_states(probs, n=20_000)
    p0_est = float((states[:, 0] == 1).float().mean())
    p1_est = float((states[:, 1] == 1).float().mean())
    assert abs(p0_est - 0.1) < 0.01
    assert abs(p1_est - 0.7) < 0.01


def test_eval_batch_serial_matches_direct_sfun():
    probs, sfun, row_names = _binomial_tail_problem(n_var=8, p_fail=0.2, fail_threshold=4)
    ss.set_worker_state(sfun, row_names)
    states = ss.sample_prior_states(probs, n=50)
    fvals, sys_sts = ss.eval_batch(states, sfun, row_names, pool=None)

    # Manually verify each one
    for i in range(50):
        cst = {row_names[k]: int(states[i, k]) for k in range(len(row_names))}
        f, s, _ = sfun(cst)
        assert fvals[i] == pytest.approx(f)
        assert sys_sts[i] == s


# ---------------------------------------------------------------------------
# CWM-H chain tests
# ---------------------------------------------------------------------------
def test_cwmh_chain_keeps_samples_in_failure_set():
    """Every accepted sample must satisfy s >= threshold (sticky-prior is rejection-based)."""
    torch.manual_seed(0)
    np.random.seed(0)
    probs, sfun, row_names = _binomial_tail_problem(n_var=20, p_fail=0.1, fail_threshold=4)
    ss.set_worker_state(sfun, row_names)

    # Manufacture seeds: states with at least 4 failed components
    seeds_list = []
    while len(seeds_list) < 10:
        s = ss.sample_prior_states(probs, n=100)
        for j in range(s.shape[0]):
            n_failed = int((s[j] == 1).sum())
            if n_failed >= 4:
                seeds_list.append(s[j])
                if len(seeds_list) == 10:
                    break
    seeds = torch.stack(seeds_list)

    seed_fvals, seed_sys = ss.eval_batch(seeds, sfun, row_names, pool=None)
    threshold_s = 4.0          # severity_sign=+1, so s = fval

    states_out, fvals_out, sys_out, n_calls = ss.cwmh_chain(
        seed_states=seeds,
        seed_fvals=seed_fvals,
        seed_sys_sts=seed_sys,
        probs=probs,
        threshold_s=threshold_s,
        severity_sign=+1,
        chain_length=10,
        sfun=sfun,
        row_names=row_names,
        n_flip_mean=2.0,
        pool=None,
    )

    # Every sample in the chain output must have fval >= 4
    assert (fvals_out >= threshold_s).all(), \
        f"chain produced samples below threshold: min={fvals_out.min()}"
    # Sanity: chain output has the right size
    assert states_out.shape == (10 * 10, 20)
    assert n_calls == 10 * (10 - 1)


def test_cwmh_chain_uniform_acceptance_at_trivial_threshold():
    """If threshold is below the min severity in the prior, all proposals accept."""
    torch.manual_seed(1)
    probs, sfun, row_names = _binomial_tail_problem(n_var=10, p_fail=0.5, fail_threshold=0)
    ss.set_worker_state(sfun, row_names)

    seeds = ss.sample_prior_states(probs, n=8)
    seed_fvals, seed_sys = ss.eval_batch(seeds, sfun, row_names, pool=None)

    # Threshold s >= -1 is satisfied by every state (n_failed >= 0)
    states_out, fvals_out, sys_out, _ = ss.cwmh_chain(
        seed_states=seeds, seed_fvals=seed_fvals, seed_sys_sts=seed_sys,
        probs=probs, threshold_s=-1.0, severity_sign=+1,
        chain_length=5, sfun=sfun, row_names=row_names,
        n_flip_mean=5.0, pool=None,
    )
    assert (fvals_out >= -1).all()
    # With n_flip_mean=5 and trivial acceptance, the chain *should* visit
    # different states across steps (extremely unlikely to remain at seeds).
    final_block = states_out[8 * 4 : 8 * 5]
    assert not torch.equal(final_block, seeds), \
        "trivial-threshold chain failed to move from seeds"


# ---------------------------------------------------------------------------
# Outer subset_sim_search tests
# ---------------------------------------------------------------------------
def test_subset_sim_finds_failures_on_rare_event():
    """SuS should reach the failure boundary for a moderately rare event."""
    torch.manual_seed(7)
    np.random.seed(7)
    n_var, p_fail, k = 20, 0.05, 8
    probs, sfun, row_names = _binomial_tail_problem(n_var, p_fail, fail_threshold=k)

    ss.set_worker_state(sfun, row_names)
    res = ss.subset_sim_search(
        probs=probs, sfun=sfun, row_names=row_names, sys_surv_st=1,
        n_per_level=1000, p0=0.1, max_levels=8,
        severity_sign=+1, n_flip_mean=3.0, pool=None, verbose=False,
    )

    assert res["terminated_by"] in ("failure_boundary", "degenerate_threshold")
    assert res["n_levels"] >= 1
    # All returned failed_states must actually be failures
    if res["failed_states"].numel() > 0:
        for j in range(res["failed_states"].shape[0]):
            cst = {row_names[i]: int(res["failed_states"][j, i]) for i in range(n_var)}
            _, sys_st, _ = sfun(cst)
            assert sys_st < 1
    # We expect SuS to find at least one failure for this regime
    assert res["failed_states"].shape[0] > 0


def test_subset_sim_severity_sign_negative():
    """Mirror the binomial-tail problem so 'low fval = bad' and severity_sign=-1."""
    torch.manual_seed(11)
    np.random.seed(11)
    n_var = 12
    probs = torch.tensor([[0.9, 0.1]] * n_var, dtype=torch.float32)

    # 'goodness' = number of working components; failure when goodness <= 5
    def sfun(comps_st):
        n_working = sum(1 for v in comps_st.values() if v == 0)
        fval = float(n_working)
        sys_st = 1 if n_working > 5 else 0
        return fval, sys_st, None

    row_names = [f"c{i}" for i in range(n_var)]
    ss.set_worker_state(sfun, row_names)

    res = ss.subset_sim_search(
        probs=probs, sfun=sfun, row_names=row_names, sys_surv_st=1,
        n_per_level=500, p0=0.1, max_levels=6,
        severity_sign=-1, n_flip_mean=3.0, pool=None, verbose=False,
    )
    assert res["terminated_by"] in ("failure_boundary", "degenerate_threshold")
    if res["failed_states"].numel() > 0:
        # Verify failed states are real failures
        for j in range(min(20, res["failed_states"].shape[0])):
            cst = {row_names[i]: int(res["failed_states"][j, i]) for i in range(n_var)}
            _, sys_st, _ = sfun(cst)
            assert sys_st < 1


def test_subset_sim_degenerate_threshold_guard():
    """If the severity is constant the chain can't progress; SuS must halt cleanly."""
    torch.manual_seed(3)
    np.random.seed(3)
    n_var = 6
    probs = torch.tensor([[0.5, 0.5]] * n_var, dtype=torch.float32)

    # Constant severity 0 — every state has fval = 0
    def sfun(comps_st):
        return 0.0, 1, None  # always survival

    row_names = [f"c{i}" for i in range(n_var)]
    ss.set_worker_state(sfun, row_names)

    res = ss.subset_sim_search(
        probs=probs, sfun=sfun, row_names=row_names, sys_surv_st=1,
        n_per_level=200, p0=0.1, max_levels=5,
        severity_sign=+1, pool=None, verbose=False,
    )
    assert res["terminated_by"] == "degenerate_threshold"
    assert res["failed_states"].shape[0] == 0


def test_subset_sim_probability_estimate_order_of_magnitude():
    """
    SuS conditional probability estimate should be in the right ballpark.
    P(F) ≈ p0^(M-1) * (n_fail_final / N).

    We use a relatively easy regime (target P ~ 1e-3) and accept a generous
    tolerance — this is a smoke check, not a precision benchmark.
    """
    torch.manual_seed(13)
    np.random.seed(13)
    n_var, p_fail, k = 30, 0.1, 7
    probs, sfun, row_names = _binomial_tail_problem(n_var, p_fail, fail_threshold=k)
    p_true = _true_binomial_tail_prob(n_var, p_fail, k)
    assert 1e-4 < p_true < 1e-1   # sanity on the test setup itself

    ss.set_worker_state(sfun, row_names)
    res = ss.subset_sim_search(
        probs=probs, sfun=sfun, row_names=row_names, sys_surv_st=1,
        n_per_level=1000, p0=0.1, max_levels=6,
        severity_sign=+1, n_flip_mean=3.0, pool=None, verbose=False,
    )

    M = res["n_levels"]
    n_fail_final = int((res["final_sys_sts"] < 1).sum())
    p_est = (0.1 ** max(M - 1, 0)) * n_fail_final / 1000

    # Order-of-magnitude check (within ~factor 10)
    if p_est > 0:
        ratio = p_est / p_true
        assert 0.1 < ratio < 10, f"SuS p_est={p_est:.3e} vs p_true={p_true:.3e}"
