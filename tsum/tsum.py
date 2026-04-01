import torch
import itertools
import operator
from itertools import product, combinations
from math import prod
import math
from decimal import Decimal
import numpy as np
import os, json, time
from typing import Callable, Dict, Any, List, Optional, Tuple, Sequence, Iterable, Union
from torch import Tensor
import psutil

import random
import multiprocessing as mp
from collections import deque

import tsum

# ---- Shared state for parallel worker processes (inherited via fork) ----
_MP_SFUN = None
_MP_SYS_SURV_ST = None
_MP_N_STATE = None


def _minimize_one_unknown(args):
    """
    Worker function for parallel minimization of unknown samples.
    Accesses module-level shared state set before the pool is created.
    """
    comps_st_test, fval = args
    sfun = _MP_SFUN
    sys_surv_st = _MP_SYS_SURV_ST
    n_state = _MP_N_STATE

    fval, sys_st, min_comps_st0 = sfun(comps_st_test)
    if min_comps_st0 is None:
        min_comps_st0 = comps_st_test.copy()
    elif isinstance(next(iter(min_comps_st0.values())), tuple):
        min_comps_st0 = {k: v[1] for k, v in min_comps_st0.items()}

    if sys_st >= sys_surv_st:
        min_comps_st, info = minimise_surv_states_random(
            min_comps_st0, sfun, sys_surv_st=sys_surv_st, fval=fval)
        fval = info.get('final_sys_state', fval)
    else:
        min_comps_st, info = minimise_fail_states_random(
            min_comps_st0, sfun, max_state=n_state - 1,
            sys_fail_st=sys_surv_st - 1, fval=fval)
        fval = info.get('final_sys_state', fval)

    return min_comps_st, sys_st, fval

# For use in mixted sorting 
try:
    import numpy as np
    _NUMPY_NUM = (np.integer, np.floating)
except Exception:
    _NUMPY_NUM = tuple()


def get_min_fail_comps_st(comps_st, max_st, sys_fail_st):
    """
    Get the minimal failing component states from a given state,
    by recording components in comps_st != max_st

    Args:
        comps_st (dict): {comp_name: state (int)}
        max_st (int): the highest state
        sys_fail_st (int): the system failure state

    Returns:
        (dict): {comp_name: ('comparison_operator', state (int))}

    """
    min_comps_st = {k: ('<=', v) for k, v in comps_st.items() if v < max_st}
    min_comps_st['sys'] = ('<=', sys_fail_st)
    return min_comps_st


def get_min_surv_comps_st(comps_st, sys_surv_st):
    """
    Get the minimal surviving component states from a given state,
    by recording components in comps_st != max_st

    Args:
        comps_st (dict): {comp_name: state (int)}
        sys_surv_st (int): the system survival state

    Returns:
        (dict): {comp_name: ('comparison_operator', state (int))}

    """
    min_comps_st = {k: ('>=', v) for k, v in comps_st.items() if v > 0}
    min_comps_st['sys'] = ('>=', sys_surv_st)
    return min_comps_st


def minimise_surv_states_random(
    comps_st: Dict[str, int],
    sfun: Callable[[Dict[Any, int]], Tuple[Any, Tuple[str, int], Dict[Any, int]]],
    sys_surv_st: int,
    *,
    fval: Optional[Any] = None,
    min_state: int = 0,
    step: int = 1,
    seed: Optional[int] = None,
    exclude_keys: Iterable[str] = ("sys",)
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Random greedy reduction of component states.

    Algorithm (given a random permutation of components):
      - Try lowering each component by `step` (e.g., 1).
      - Call sfun(modified_state).
        Expect sfun to return a tuple where the 2nd element is int that represents a system state.
      - If status >= sys_surv_st: keep the lowered value and continue cycling.
        If the component reaches `min_state`, remove it (can't lower further).
      - If status < sys_fail_st: revert the change and remove that component (no further attempts).

    Stops when all components have been removed from the candidate pool.

    Returns:
      final_state, info
        - final_state: dict of the minimized states.
        - info: {
            'permutation': [...],
            'removed_on_failure': [comp,...],
            'hit_min_state': [comp,...],
            'attempts': int,
          }
    """
    rng = random.Random(seed)

    # Work on a (shallow) copy; do NOT mutate caller's dict (value int is immutable)
    state = dict(comps_st)

    # Build candidate component key deque from a random permutation
    candidates = [k for k, v in state.items()
                  if k not in set(exclude_keys) and isinstance(v, int) and v > min_state]
    rng.shuffle(candidates)
    dq = deque(candidates)

    removed_on_failure = []
    hit_min_state = []
    attempts = 0

    while dq:
        comp = dq[0]

        # If already at/below min_state, remove and continue
        #if state.get(comp, min_state) <= min_state: # state[comp] always works
        if state[comp] <= min_state: # state[comp] always works
            dq.popleft()
            hit_min_state.append(comp)
            continue

        prev = state[comp]
        fval_prev = fval
        state[comp] = prev - step
        attempts += 1

        # Expect sfun to return (value, 's'/'f', info) or similar
        try:
            fval, status, _ = sfun(state)
        except Exception as e:
            # If your sfun has a different signature, surface the error clearly
            state[comp] = prev  # revert
            fval = fval_prev
            dq.popleft()
            removed_on_failure.append(comp)
            continue

        if status >= sys_surv_st:
            # Keep lowered value
            if state[comp] <= min_state:
                dq.popleft()
                hit_min_state.append(comp)
            else:
                dq.rotate(-1)  # move to back; try again later
        else:
            # Revert and remove from further consideration
            state[comp] = prev
            fval = fval_prev
            dq.popleft()
            removed_on_failure.append(comp)

    info = {
        'permutation': candidates,
        'removed_on_failure': removed_on_failure,
        'hit_min_state': hit_min_state,
        'attempts': attempts,
        'final_state': state,
        'final_sys_state': fval
    }

    min_rule = get_min_surv_comps_st(state, sys_surv_st)

    return min_rule, info


def minimise_fail_states_random(
    comps_st: Dict[str, int],
    sfun,
    sys_fail_st: int,
    max_state: int,
    *,
    fval: Optional[Any] = None,
    step: int = 1,
    seed: Optional[int] = None,
    exclude_keys: Iterable[str] = ("sys",)
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Random greedy reduction of component states.

    Algorithm (given a random permutation of components):
      - Try increasing each component by `step` (e.g., 1).
      - Call sfun(modified_state).
        Expect sfun to return a tuple where the 2nd element is an int representing a system state.
      - If status <= sys_fail_st: keep the increased value and continue cycling.
        If the component reaches `max_state`, remove it (can't increase further).
      - If status > sys_fail_st: revert the change and remove that component (no further attempts).

    Stops when all components have been removed from the candidate pool.

    Returns:
      final_state, info
        - final_state: dict of the minimized states.
        - info: {
            'permutation': [...],
            'removed_on_failure': [comp,...],
            'hit_min_state': [comp,...],
            'attempts': int,
            'final_state': {comp: state,...}
          }
    """
    rng = random.Random(seed)

    # Work on a copy; do NOT mutate caller's dict
    state = dict(comps_st)

    # Build candidate deque from a random permutation
    candidates = [k for k, v in state.items()
                  if k not in set(exclude_keys) and isinstance(v, int) and v < max_state]
    rng.shuffle(candidates)
    dq = deque(candidates)

    removed_on_survival = []
    hit_min_state = []
    attempts = 0

    while dq:
        comp = dq[0]

        # If already at/below min_state, remove and continue
        if state.get(comp, max_state) >= max_state:
            dq.popleft()
            hit_min_state.append(comp)
            continue

        prev = state[comp]
        fval_prev = fval
        state[comp] = prev + step
        attempts += 1

        # Expect sfun to return (value, 's'/'f', info) or similar
        try:
            fval, status, _ = sfun(state)
        except Exception as e:
            # If your sfun has a different signature, surface the error clearly
            state[comp] = prev  # revert
            fval = fval_prev
            dq.popleft()
            removed_on_survival.append(comp)
            continue

        if status <= sys_fail_st:
            # Keep increased value
            if state[comp] >= max_state:
                dq.popleft()
                hit_min_state.append(comp)
            else:
                dq.rotate(-1)  # move to back; try again later
        else:
            # Revert and remove from further consideration
            state[comp] = prev
            fval = fval_prev
            dq.popleft()
            removed_on_survival.append(comp)

    info = {
        'permutation': candidates,
        'removed_on_survival': removed_on_survival,
        'hit_min_state': hit_min_state,
        'attempts': attempts,
        'final_state': state,
        'final_sys_state': fval
    }

    min_rule = get_min_fail_comps_st(state, max_state, sys_fail_st)

    return min_rule, info


def from_rule_dict_to_mat(rule_dict, row_names, max_st):
    """
    Convert a rule dictionary to a matrix representation.

    Args:
        rule_dict (dict): {name: ('comparison_operator', state (int))}
        row_names (list): list of component names associated with each row in order
        max_st (int): the highest state

    Returns:
        mat (list): binary matrix with shape (n_comp, max_st)

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mat = torch.zeros((len(row_names), max_st), dtype=torch.int32, device=device)

    for row, name in enumerate(row_names):  
        if name in rule_dict:
            op, state = rule_dict[name]
            if op == '<=':
                mat[row, :state + 1] = 1
            elif op == '<':
                mat[row, :state] = 1
            elif op == '>=':
                mat[row, state:] = 1
            elif op == '>':
                mat[row, state + 1:] = 1
            elif op == '==':
                mat[row, state] = 1
        else:
            mat[row, :] = 1

    return mat

def from_Bbound_to_comps_st(Bbound, row_names):
    """
    Extracts the index of the first non-zero state for each component (ignoring the system row).

    Args:
        Bbound (Tensor): shape (n_var, n_state)
        row_names (list): list of variable names including system

    Returns:
        comps_st (dict): {component_name: state_index}
    """
    n_var, n_state = Bbound.shape

    comps_st = {}
    for i in range(n_var):
        row = Bbound[i]
        nz = torch.nonzero(row, as_tuple=False)
        if len(nz) > 0:
            comps_st[row_names[i]] = int(nz[0])
        else:
            comps_st[row_names[i]] = None  # or raise an error

    return comps_st

def get_branches_cap_branches(B1, B2, batch_size=64):
    """
    Memory-efficient intersection of branches with batching over the larger tensor (B1 or B2).
    Inputs:
        B1: (n_br1, n_var, n_state)
        B2: (n_br2, n_var, n_state)
    Returns:
        Bnew: (n_valid, n_var, n_state)
    """
    device = B1.device
    n_br1, n_var, n_state = B1.shape
    n_br2 = B2.shape[0]
    results = []

    if n_br1 >= n_br2:
        # Batch over B1
        for start in range(0, n_br1, batch_size):
            end = min(start + batch_size, n_br1)
            B1_batch = B1[start:end]                    # (batch_size, n_var, n_state)

            B1_exp = B1_batch.unsqueeze(1)              # (batch_size, 1, n_var, n_state)
            B2_exp = B2.unsqueeze(0)                    # (1, n_br2, n_var, n_state)
            Bnew = B1_exp & B2_exp                      # (batch_size, n_br2, n_var, n_state)
            Bnew = Bnew.view(-1, n_var, n_state)

            # Filter invalid
            invalid_mask = (Bnew == 0).all(dim=2)
            keep_mask = ~invalid_mask.any(dim=1)
            Bnew = Bnew[keep_mask]

            results.append(Bnew)
    else:
        # Batch over B2
        for start in range(0, n_br2, batch_size):
            end = min(start + batch_size, n_br2)
            B2_batch = B2[start:end]

            B1_exp = B1.unsqueeze(1)                    # (n_br1, 1, n_var, n_state)
            B2_exp = B2_batch.unsqueeze(0)              # (1, batch_size, n_var, n_state)
            Bnew = B1_exp & B2_exp                      # (n_br1, batch_size, n_var, n_state)
            Bnew = Bnew.view(-1, n_var, n_state)

            invalid_mask = (Bnew == 0).all(dim=2)
            keep_mask = ~invalid_mask.any(dim=1)
            Bnew = Bnew[keep_mask]

            results.append(Bnew)

    if results:
        return torch.cat(results, dim=0)
    else:
        return torch.empty((0, n_var, n_state), dtype=B1.dtype, device=device)

def get_complementary_events(mat):
    """
    Given a (n_vars, n_state) matrix with the last row as the system event,
    generate a set of complementary logical events (one per component).

    Returns:
        Bnew: (n_comps_kept, n_vars, n_state)
    """
    n_vars, n_state = mat.shape

    # Prepare output tensor
    B = torch.ones((n_vars, n_vars, n_state), dtype=mat.dtype, device=mat.device)

    # Broadcast mat for all i
    mat_exp = mat.unsqueeze(0).expand(n_vars, n_vars, n_state)

    # Create lower-triangular mask to copy rows before i
    mask = torch.arange(n_vars, device=mat.device).unsqueeze(0) < torch.arange(n_vars, device=mat.device).unsqueeze(1)  # (n_vars, n_vars)
    mask = mask.unsqueeze(-1).expand(-1, -1, n_state)  # (n_vars, n_vars, n_state)
    B[mask] = mat_exp[mask]  # copy rows before i

    # Flip row i in each batch
    flip_mask = torch.eye(n_vars, dtype=torch.bool, device=mat.device).unsqueeze(-1).expand(-1, -1, n_state)  # (n_vars, n_vars, n_state)
    B[:n_vars, :n_vars][flip_mask] = 1 - mat_exp[:n_vars, :n_vars][flip_mask]

    # Remove combinations where any row (excluding system) is all-zero across states
    invalid_mask = (B[:, :-1, :] == 0).all(dim=2)  # shape: (n_vars, n_vars)
    keep_mask = ~invalid_mask.any(dim=1)          # shape: (n_vars,)
    Bnew = B[keep_mask]

    return Bnew

def get_branch_probs(tensor, prob):
    """
    Computes the probability of each branch given a binary event tensor and state probabilities.

    Args:
        tensor: (n_br, n_var, n_state) - binary indicator of active states per variable per branch
        prob:   (n_var, n_state)   - probability per state for each component variable

    Returns:
        Bprob: (n_br,) - probability per branch
    """
    n_br, n_var, n_state = tensor.shape
    device = tensor.device

    # Expand to match tensor: (n_br, n_comps, n_state)
    prob_exp = prob.unsqueeze(0).expand(n_br, -1, -1)

    # Element-wise multiplication and summing across states
    prob_selected = tensor * prob_exp  # (n_br, n_comps, n_state)
    prob_per_var = prob_selected.sum(dim=2)  # (n_br, n_comps)
    Bprob = prob_per_var.prod(dim=1)  # (n_br,)

    return Bprob

import torch

def get_boundary_branches(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute boundary branches for each input branch.

    Input:
        tensor: (n_vars, n_state)  OR  (n_br, n_vars, n_state)
                int/bool tensor with 0/1 entries.

                - n_vars includes the system row as the LAST row.
                - For each component row, the active state(s) are 1s along n_state.
                  We pick:
                    * 'lower' boundary: first active state (min index)
                    * 'upper' boundary: last  active state (max index)

    Output:
        (2, n_vars, n_state)             if input was (n_vars, n_state)
        (2*n_br, n_vars, n_state)        if input was (n_br, n_vars, n_state)

        The last row (system row) is set to all 1s in both upper and lower outputs.
    """
    assert tensor.ndim in (2, 3), "Input must be 2D (n_vars,n_state) or 3D (n_br,n_vars,n_state)"

    # Normalize to 3D (batch of branches)
    squeeze_back = (tensor.ndim == 2)
    if squeeze_back:
        x = tensor.unsqueeze(0)  # (1, n_vars, n_state)
    else:
        x = tensor               # (n_br, n_vars, n_state)

    n_br, n_vars, n_state = x.shape
    # n_comps = n_vars - 1 # OBSOLETE: system row is now excluded from input
    n_comps = n_vars

    # Work only on component rows (exclude final system row)
    comp = x[:, :n_comps, :]             # (n_br, n_comps, n_state)
    mask = comp.bool()

    # First active state index per component
    first_hit = (mask.float().cumsum(dim=-1) == 1).float()
    first_idx = first_hit.argmax(dim=-1)  # (n_br, n_comps)

    # Last active state index per component
    rev = torch.flip(mask, dims=[-1])
    last_hit = (rev.float().cumsum(dim=-1) == 1).float()
    last_idx = last_hit.argmax(dim=-1)
    last_idx = (n_state - 1) - last_idx   # (n_br, n_comps)

    # (Optional) If a row has no 1s at all, both argmax above return 0.
    # If you want to *suppress* placing a 1 in such rows, detect and skip:
    # has_one = mask.any(dim=-1)                              # (n_br, n_comps)
    # first_idx = torch.where(has_one, first_idx, -1)         # -1 will be ignored by scatter_
    # last_idx  = torch.where(has_one,  last_idx,  -1)

    # Build lower/upper with one-hot at first/last indices
    lower = torch.zeros_like(comp)
    upper = torch.zeros_like(comp)

    lower.scatter_(-1, first_idx.unsqueeze(-1), 1)
    upper.scatter_(-1, last_idx.unsqueeze(-1), 1)

    # Stack branches: [upper; lower] along branch dimension
    out = torch.cat([upper, lower], dim=0)   # (2*n_br, n_vars, n_state)

    # Squeeze back if original was 2D: return shape (2, n_vars, n_state)
    return out if not squeeze_back else out.view(2, n_vars, n_state)


# FIXME: unused
def get_boundary_rules(tensor):
    n_br, n_vars, n_state = tensor.shape
    #n_comps = n_vars - 1 # exclude system event (last row) <- OUTDATED: system row is now excluded from input
    n_comps = n_vars

    comp_tensor = tensor[:, :n_comps, :]  # (n_br, n_comps, n_state)

    # Create boolean mask of active entries
    mask = comp_tensor.bool()  # (n_br, n_comps, n_state)

    # Get first and last nonzero indices
    first_idx = mask.float().cumsum(dim=2)
    first_idx = (first_idx == 1).float()
    first_idx = first_idx.argmax(dim=2)  # (n_br, n_comps)

    # Reverse to find last
    reversed_mask = torch.flip(mask, dims=[2])
    last_idx = reversed_mask.float().cumsum(dim=2)
    last_idx = (last_idx == 1).float()
    last_idx = last_idx.argmax(dim=2)
    last_idx = n_state - 1 - last_idx  # reverse indices

    # Build upper and lower tensors
    ###### only this part is different from get_boundary_branches #####
    state_idx = torch.arange(n_state, device=last_idx.device).view(1, 1, -1).expand(n_br, n_comps, -1)
    upper = (state_idx >= last_idx.unsqueeze(-1)).to(tensor.dtype)
    lower = (state_idx <= first_idx.unsqueeze(-1)).to(tensor.dtype)
    ####################################################################

    # Append system row of all 1s 
    #system = torch.ones((n_br, 1, n_state), dtype=tensor.dtype, device=tensor.device)

    #B_upper = torch.cat([upper, system], dim=1)
    B_upper = upper
    #B_lower = torch.cat([lower, system], dim=1)
    B_lower = lower

    return torch.cat([B_upper, B_lower], dim=0)  # shape: (2*n_br, n_vars, n_state)

# FIXME: ununsed
def is_intersect(events1, events2):
    """
    Determine whether each event in events1 intersects with any event in events2.

    Args:
        events1: (n_event1, n_vars, n_state)
        events2: (n_event2, n_vars, n_state)

    Returns:
        labels: (n_event1,) boolean tensor
    """
    n_event1, n_vars, n_state = events1.shape
    n_event2, _, _ = events2.shape

    # Expand for broadcasting
    events1_exp = events1.unsqueeze(1).expand(-1, n_event2, -1, -1)
    events2_exp = events2.unsqueeze(0).expand(n_event1, -1, -1, -1)

    # Compute intersection and check if any is non-zero per pair
    intersect = events1_exp & events2_exp  # logical AND
    is_empty = (intersect == 0).all(dim=3).any(dim=2)  # shape: (n_event1, n_event2)
    labels = ~is_empty.all(dim=1)  # if any intersected, mark True

    return labels

def is_subset(mat, tensor):
    """
    Checks if:
      1. `mat` is a subset of any of the events in `tensor`, and
      2. Any of the events in `tensor` is a subset of `mat`.

    Args:
        mat: Tensor of shape (n_var, n_state)
        tensor: Tensor of shape (n_event, n_var, n_state)

    Returns:
        is_mat_subset: bool
        is_tensor_subset: BoolTensor of shape (n_event,)
    """
    n_event, n_var, n_state = tensor.shape
    mat_e = mat.unsqueeze(0).expand(n_event, -1, -1)  # (n_event, n_var, n_state)

    intersect = mat_e & tensor  # (n_event, n_var, n_state)

    is_mat_subset = torch.any(torch.all(mat_e == intersect, dim=(1, 2))).item()
    is_tensor_subset = torch.all(tensor == intersect, dim=(1, 2))  # shape (n_event,)

    return bool(is_mat_subset), is_tensor_subset

import torch
from math import prod

@torch.no_grad()
def find_first_nonempty_combination(Rcs, batch_size=65536, verbose=False):
    """
    Rcs: list[Tensor] with shapes (n_i, n_vars, n_state), same (n_vars, n_state)
    Order: increasing sum of tuple indices, then lexicographic within that sum.
    Returns: (selected_mat: (n_vars, n_state), idx_tuple) or (None, None)
    """
    assert len(Rcs) > 0
    device = Rcs[0].device
    n_vars, n_state = Rcs[0].shape[1:]
    ns = torch.tensor([r.shape[0] for r in Rcs], device=device, dtype=torch.long)
    k = len(ns)
    assert all((r.device == device and r.shape[1]==n_vars and r.shape[2]==n_state) for r in Rcs)

    # total combinations and linear-index strides (mixed radix, right-to-left)
    n_combs = int(torch.prod(ns).item())
    if n_combs == 0:
        return None

    strides = torch.ones_like(ns)
    if k > 1:
        strides[:-1] = torch.cumprod(ns.flip(0)[:-1], dim=0).flip(0)  # lex rank weights too

    # maximum possible sum level
    max_sum = int((ns - 1).sum().item())

    # scan sum shells s = 0..max_sum
    for s in range(max_sum + 1):
        if verbose:
            print(f"[sum={s}] scanning...")
        start = 0

        best_lex_rank = None
        best_sel_mat = None
        best_tuple = None
        best_global_idx = None

        while start < n_combs:
            end = min(start + batch_size, n_combs)

            # linear indices (GPU)
            lin = torch.arange(start, end, device=device, dtype=torch.long)

            # decode to tuples (batch, k)
            idx = (lin[:, None] // strides) % ns

            # filter rows at the current sum level
            sum_mask = (idx.sum(dim=1) == s)
            if sum_mask.any():
                idx_s = idx[sum_mask]
                # gather the needed rows only (saves compute)
                mats = [r[idx_s[:, i]] for i, r in enumerate(Rcs)]
                mat = torch.stack(mats, dim=0).prod(dim=0)  # (batch_s, n_vars, n_state)

                # non-empty check (your original rule)
                is_empty = (mat == 0).all(dim=2).any(dim=1)
                valid = ~is_empty

                if valid.any():
                    # lexicographic rank within this sum shell
                    lex_rank = (idx_s * strides).sum(dim=1)  # dot with strides
                    # among valid, choose min lex
                    lex_rank_valid = lex_rank.clone()
                    # mask out invalid by setting to +inf
                    lex_rank_valid[~valid] = torch.iinfo(torch.int64).max

                    # candidate in this batch
                    batch_min_lex, batch_pos = torch.min(lex_rank_valid, dim=0)
                    if batch_min_lex != torch.iinfo(torch.int64).max:
                        # global best within this sum s (merge across batches)
                        if (best_lex_rank is None) or (batch_min_lex < best_lex_rank):
                            best_lex_rank = batch_min_lex
                            best_sel_mat = mat[batch_pos]              # (n_vars, n_state)
                            best_tuple = tuple(int(v) for v in idx_s[batch_pos].tolist())
                            best_global_idx = int((idx_s[batch_pos] * strides).sum().item())

            start = end

        if best_sel_mat is not None:
            if verbose:
                print(f"Selected index: {best_tuple} (sum={s}, lex_rank={int(best_lex_rank)}, lin={best_global_idx})")
            return best_sel_mat

    return None


# FIXME: unused
def sum_sorted_tuples_limited(max_vals):
    """
    Generate all tuples of non-negative integers with len=max_vals,
    where each element i ≤ max_vals[i],
    ordered by increasing sum, then lexicographically.
    
    Args:
        max_vals (list or tuple): list of maximum values per position.
    
    Yields:
        tuple of ints
    """
    n = len(max_vals)
    sum_level = 0
    while True:
        found = False
        for t in itertools.product(*(range(v+1) for v in max_vals)):
            if sum(t) == sum_level:
                yield t
                found = True
        if not found:
            break  # no more combinations possible
        sum_level += 1

# FIXME: unused
def merge_branches(B):
    "Use hashing for computational efficiency"

    is_merge = True

    while is_merge:
        B_com = bit_compress(B)
        groups_by_col = groups_by_column_remhash_dict(B_com)
        merges = plan_merges(groups_by_col, B.shape[0])
        B, _ = apply_merges(B, merges)

        is_merge = any(len(g) > 1 for g in groups_by_col)

    return B

# FIXME: unused
def merge_branches_old(B, batch_size=100_000):
    device = B.device
    dtype = B.dtype

    B = B.clone()
    changed = True

    while changed:
        changed = False
        n_br, n_comp, n_state = B.shape
        keep_mask = torch.ones(n_br, dtype=torch.bool, device=device)
        new_branches = []

        # Generate all i < j combinations
        all_pairs = list(combinations(range(n_br), 2))
        total_pairs = len(all_pairs)

        used = torch.zeros(n_br, dtype=torch.bool, device=device)

        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)
            idx_i, idx_j = zip(*all_pairs[start:end])
            idx_i = torch.tensor(idx_i, device=device)
            idx_j = torch.tensor(idx_j, device=device)

            bi = B[idx_i]  # (n_pair, n_comp, n_state)
            bj = B[idx_j]

            # Step 1: Compare along components to count differing rows
            diffs = (bi != bj).any(dim=2)  # (n_pair, n_comp)
            num_diff_rows = diffs.sum(dim=1)  # (n_pair,)
            one_diff_mask = num_diff_rows == 1

            if one_diff_mask.sum() == 0:
                continue  # no valid pairs in this batch

            valid_idx_i = idx_i[one_diff_mask]
            valid_idx_j = idx_j[one_diff_mask]
            valid_diffs = diffs[one_diff_mask]
            valid_bi = bi[one_diff_mask]
            valid_bj = bj[one_diff_mask]

            diff_row_idx = valid_diffs.float().argmax(dim=1)  # (n_valid_pairs,)

            # Extract differing rows
            ri = torch.stack([valid_bi[k, diff_row_idx[k]] for k in range(len(diff_row_idx))])
            rj = torch.stack([valid_bj[k, diff_row_idx[k]] for k in range(len(diff_row_idx))])

            disjoint_mask = (ri & rj).sum(dim=1) == 0
            if disjoint_mask.sum() == 0:
                continue

            final_merge_indices = []
            for k in range(disjoint_mask.size(0)):
                if not disjoint_mask[k]:
                    continue
                i = valid_idx_i[k].item()
                j = valid_idx_j[k].item()
                if used[i] or used[j]:
                    continue
                used[i] = True
                used[j] = True
                final_merge_indices.append(k)

            if not final_merge_indices:
                continue

            changed = True
            final_merge_indices = torch.tensor(final_merge_indices, device=device)

            disjoint_i = valid_idx_i[final_merge_indices]
            disjoint_j = valid_idx_j[final_merge_indices]
            disjoint_bi = B[disjoint_i]
            disjoint_bj = B[disjoint_j]
            disjoint_diff_idx = diff_row_idx[disjoint_mask][final_merge_indices]

            for i in range(disjoint_i.size(0)):
                merged = disjoint_bi[i].clone()
                merged[disjoint_diff_idx[i]] = disjoint_bi[i][disjoint_diff_idx[i]] | disjoint_bj[i][disjoint_diff_idx[i]]
                new_branches.append(merged)

        keep_mask[used] = False
        if new_branches:
            B = torch.cat([B[keep_mask], torch.stack(new_branches)], dim=0)

    return B

def get_complementary_events_nondisjoint(mat: torch.Tensor) -> torch.Tensor:
    """
    Given a (n_vars, n_state) matrix with the last row as the system event,
    generate a set of complementary logical events by flipping each row.
    NOTE: The resulted events are not disjoint.

    Returns:
        Bnew: (n_events_kept, n_vars, n_state)
    """
    n_vars, n_state = mat.shape

    # Prepare output tensor
    B = torch.ones((n_vars, n_vars, n_state), dtype=mat.dtype, device=mat.device)

    # Flip row i in batch i
    idx = torch.arange(n_vars, device=mat.device)
    if mat.dtype == torch.bool:
        B[idx, idx, :] = ~mat[idx, :]
    else:
        # assumes binary in {0,1}; works for float or int tensors
        B[idx, idx, :] = 1 - mat[idx, :]

    # Remove combinations where any row (excluding system) is all-zero across states
    invalid_mask = (B == 0).all(dim=2)  # shape: (n_vars, n_vars)
    keep_mask = ~invalid_mask.any(dim=1)          # shape: (n_vars,)
    Bnew = B[keep_mask]

    return Bnew

def bit_compress(B: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor B of shape (n, m, k) of bits {0,1}
    into an integer tensor of shape (n, m),
    where each element is sum_k 2^k * B[i,j,k].
    """
    n, m, k = B.shape
    # weights = [1, 2, 4, ..., 2^(k-1)]
    weights = (2 ** torch.arange(k, device=B.device, dtype=torch.int32))
    return (B.to(torch.int32) * weights).sum(dim=2)


def groups_by_column_remhash_dict(X: torch.Tensor):
    """
    For each column j, return groups of row indices that are identical on all
    other columns but differ on column j. Uses removable hashes + CPU dict
    (expected O(m n)) and includes a collision-guard verification.
    """
    X = X.to(torch.long)
    device = X.device
    n, m = X.shape
    out = [[] for _ in range(m)]
    if n == 0 or m == 0:
        return out

    # Two primes + per-column coefficients
    p1 = 2_147_483_629
    p2 = 2_147_483_647
    a1 = torch.arange(1, m + 1, device=device, dtype=torch.long)
    a2 = (a1 * 1315423911) % p2

    # Precompute full-row hashes
    H1 = (X * a1).sum(dim=1) % p1
    H2 = (X * a2).sum(dim=1) % p2

    for j in range(m):
        H1_wo = (H1 - X[:, j] * a1[j]) % p1
        H2_wo = (H2 - X[:, j] * a2[j]) % p2

        # skinny keys to CPU dict
        keys = torch.stack((H1_wo, H2_wo), dim=1).cpu().tolist()
        buckets = {}
        for i, k in enumerate(keys):
            buckets.setdefault((int(k[0]), int(k[1])), []).append(i)

        for rows in buckets.values():
            if len(rows) < 2:
                continue
            rows_t = torch.tensor(rows, device=device)
            vals_j = X[rows_t, j]
            # must truly differ at column j
            if torch.unique(vals_j).numel() < 2:
                continue

            # --- collision guard: check equality on all other columns ---
            Xg = X[rows_t]  # (s, m)
            same_cols = (Xg == Xg[0]).all(dim=0)  # (m,) True if all rows equal in that column
            # require all columns except j to be identical
            if bool(same_cols[torch.arange(m, device=device) != j].all()):
                out[j].append(rows_t)

    return out


def plan_merges(groups_per_col, n_rows):
    """
    groups_per_col: list where groups_per_col[j] is a list of 1D LongTensors of row indices (same device)
    n_rows: total number of rows
    returns: list of (i, k, j) merges, greedy, non-overlapping across all columns
    """
    # Track rows already used in a merge
    # Keep this on CPU bool for simplicity; adjust to CUDA if you prefer
    used = torch.zeros(n_rows, dtype=torch.bool)
    merges = []

    for j, groups in enumerate(groups_per_col):
        for g in groups:
            # Greedily pair left-to-right inside this group, skipping used rows
            # Note: keep device of g, but we only read its indices here
            # Collect unused indices in order
            unused = [int(idx) for idx in g.tolist() if not used[int(idx)].item()]
            # Pair consecutive unused
            for t in range(0, len(unused) - 1, 2):
                i, k = unused[t], unused[t+1]
                if used[i] or used[k]:
                    continue
                merges.append((i, k, j))
                used[i] = True
                used[k] = True
            # If odd count, last one is left unmatched (as you wanted)

    return merges

def apply_merges(B, merges, reducer="or"):
    """
    B: (n, m, k) tensor (CPU or CUDA)
    merges: list of (i, k, j)
    reducer: "or" (clip sum to {0,1}), "sum" (raw sum), or "max"
    Returns: (B_merged, kept_indices)
      - B_merged: tensor with merged rows; second rows in pairs are removed
      - kept_indices: 1D LongTensor mapping new rows back to old indices
    """
    device = B.device
    n, m, k = B.shape
    keep = torch.ones(n, dtype=torch.bool, device=device)

    for (i, k_idx, j) in merges:
        i = int(i); k_idx = int(k_idx); j = int(j)
        if reducer == "or":
            B[i, j] = torch.clamp(B[i, j] + B[k_idx, j], min=0, max=1)
        elif reducer == "sum":
            B[i, j] = B[i, j] + B[k_idx, j]
        elif reducer == "max":
            B[i, j] = torch.maximum(B[i, j], B[k_idx, j])
        else:
            raise ValueError("reducer must be 'or', 'sum', or 'max'")
        keep[k_idx] = False  # drop the second row of the pair

    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
    B_new = B[keep]
    return B_new, kept_indices

def sample_new_comp_st_to_test(probs, rules_mat, B=1_024, max_iters=1_000):

    device = probs.device
    n_comp, n_state = probs.shape
    #n_var = n_comp + 1  # including system event <- OUTDATED: system row is now excluded from input
    n_var = n_comp

    if len(rules_mat) == 0:
        all_samples = torch.ones((1, n_var, n_state), dtype=torch.int32, device=device)
        return all_samples[0], all_samples

    all_samples = torch.empty((0, n_var, n_state), dtype=torch.int32, device=device)

    for iter in range(max_iters):

        # Start with all-ones batch
        samples_b = torch.ones((B, n_var, n_state), dtype=torch.int32, device=device)

        # Strategy 1: The same permutation applies within a batch
        rules_ord = np.random.permutation(len(rules_mat))
        # Strategy 2: Sort the rules by their probs
        #rules_probs = get_branch_probs(rules_mat, probs)
        #rules_ord = torch.argsort(rules_probs, descending=True)

        # Sampling starts.
        for r_idx in rules_ord:

            r_mat = rules_mat[r_idx]
            r_mat_c = get_complementary_events_nondisjoint(r_mat)

            # Decide whether to sample: skip samples that already contradicts r_mat (to obtain minimal rules)
            is_sampled = torch.ones((B,), dtype=torch.bool, device=device)
            for rc1 in r_mat_c:
                flag1, flag2 = is_subset(rc1, samples_b)

                is_sampled[flag2] = False

            # Select a r_mat_c
            r_mat_c_probs = get_branch_probs(r_mat_c, probs)
            r_mat_c_probs = r_mat_c_probs / r_mat_c_probs.sum()
            idx = torch.multinomial(r_mat_c_probs, num_samples=B, replacement=True)

            # Update samples if is_sampled == True
            samples_b[is_sampled] = samples_b[is_sampled] * r_mat_c[idx[is_sampled]].squeeze(0)

        # Check if there are events with positive prob
        real_prs = get_branch_probs(samples_b, probs)

        all_samples = torch.cat((all_samples, samples_b), dim=0)

        if (real_prs > 0).any():

            x = torch.randint(0, 2, (1,)).item() # which strategy to select?
            # Strategy 1: pick the rule with the highest probability
            if x == 0:
                s_idx = torch.argmax(real_prs)
            # Strategy 2: pick the lowest probability rule
            else:
                # Replace non-positives with +inf so they don't get picked
                masked = torch.where(real_prs > 0, real_prs, torch.inf)  # (B,1)
                s_idx = torch.argmin(masked)  # scalar index into the flattened tensor
            
            sample = samples_b[s_idx] 
            
            bound_br = get_boundary_branches(sample.unsqueeze(0))
            ## decide whether to check upper or lower bound first
            x = torch.randint(0, 2, (1,)).item()
            #x = 1 # check the upper bound first
            is_b_subset, _ = is_subset(bound_br[x], rules_mat) 
            if not is_b_subset:
                return bound_br[x], all_samples
            else:
                is_a_subset, _ = is_subset(bound_br[1-x], rules_mat) 
                if not is_a_subset:
                    return bound_br[1-x], all_samples
                else:
                    Warning("Both boundary branches are subsets of the existing rules. Something's wrong.")
            
            # Strategy 2: pick the branch with the highest probability
            """samples_b = samples_b[real_prs > 0]
            samples_br = get_boundary_rules(samples_b)
            br_prs = get_branch_probs(samples_br, probs)
            br_idx = torch.argsort(br_prs, descending=True)
            for b_idx in br_idx:
                bound_br = samples_br[b_idx]
                # decide whether to check upper or lower bound first
                is_b_subset, _ = is_subset(bound_br, rules_mat) 
                if not is_b_subset:
                    return bound_br, all_samples"""

        elif iter == max_iters - 1:
            print("Max iterations reached without finding a valid sample.")
            return None, all_samples


def _check_any_subset(samples_flat, not_rules_flat, sample_chunk=10000):
    """
    Check which samples are subsets of at least one rule using matmul.

    sample ⊆ rule iff (sample & ~rule) has no 1s, i.e. sample_flat @ not_rule_flat.T == 0.

    Args:
        samples_flat: (B, D) float tensor (flattened binary samples)
        not_rules_flat: (N_rules, D) float tensor (flattened ~rules)
        sample_chunk: process this many samples at a time to bound memory

    Returns:
        (B,) bool tensor — True if sample is subset of any rule
    """
    B = samples_flat.shape[0]
    device = samples_flat.device
    result = torch.zeros(B, dtype=torch.bool, device=device)

    for start in range(0, B, sample_chunk):
        end = min(start + sample_chunk, B)
        # (chunk, D) @ (D, N_rules) → (chunk, N_rules): count of violations
        violations = samples_flat[start:end] @ not_rules_flat.T
        result[start:end] = (violations == 0).any(dim=1)

    return result


def _ensure_rules_tensor(rules, device):
    """Convert rules to a 3D tensor if given as a list."""
    if isinstance(rules, torch.Tensor):
        return rules.to(device)
    if len(rules) == 0:
        return torch.zeros((0,), device=device)
    return torch.stack([r.to(device) for r in rules])


def classify_samples(samples, survival_rules, failure_rules):
    """
    Classify samples as survival, failure, or unknown using subset checks.

    Uses batched matmul instead of per-rule loop for O(1) GPU ops regardless
    of rule count.

    Args:
        samples: (n_sample, n_var, n_state) sample tensor (binary)
        survival_rules: (n_surv, n_var, n_state) rule tensor or list
        failure_rules: (n_fail, n_var, n_state) rule tensor or list

    Returns:
        counts: dict with keys 'survival', 'failure', 'unknown'
    """
    device = samples.device
    n_sample = samples.shape[0]
    survival_rules = _ensure_rules_tensor(survival_rules, device)
    failure_rules = _ensure_rules_tensor(failure_rules, device)

    samples_flat = samples.reshape(n_sample, -1).to(dtype=torch.float16)

    # Survival check
    survival_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)
    if survival_rules.ndim == 3 and survival_rules.shape[0] > 0:
        not_surv = (~survival_rules.bool()).reshape(survival_rules.shape[0], -1).to(dtype=torch.float16)
        survival_mask = _check_any_subset(samples_flat, not_surv)

    # Failure check (only on non-survival samples)
    failure_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)
    remaining = ~survival_mask
    if failure_rules.ndim == 3 and failure_rules.shape[0] > 0 and remaining.any():
        not_fail = (~failure_rules.bool()).reshape(failure_rules.shape[0], -1).to(dtype=torch.float16)
        fail_sub = _check_any_subset(samples_flat[remaining], not_fail)
        failure_mask[remaining] = fail_sub

    counts = {
        'survival': int(survival_mask.sum().item()),
        'failure': int(failure_mask.sum().item()),
        'unknown': int((~survival_mask & ~failure_mask).sum().item())
    }
    return counts

def _sample_and_classify_on_device(args):
    """
    Sample + classify on a single GPU device. Used by multi-GPU sampling.
    Runs in a thread — GPU ops release the GIL during kernel execution.
    """
    probs_dev, n_sample, rules_surv_dev, rules_fail_dev, with_indices = args
    samples = sample_categorical(probs_dev, n_sample)
    if with_indices:
        res = classify_samples_with_indices(samples, rules_surv_dev, rules_fail_dev, return_masks=True)
    else:
        res = classify_samples(samples, rules_surv_dev, rules_fail_dev)
    return samples, res


def sample_categorical(probs, n_sample):
    """
    Sample binary event tensors from categorical distributions.

    Args:
        probs: (n_var, n_state) - probabilities per state per variable.
        n_sample: Number of samples to draw.

    Returns:
        samples: (n_sample, n_var, n_state) - one-hot encoded state selection.
    """

    device = probs.device

    n_var, n_state = probs.shape

    # Step 1: Cumulative probability
    cum_probs = torch.cumsum(probs, dim=1)  # shape (n_var, n_state)

    # Step 2: Uniform random values for each variable
    rand_vals = torch.rand(n_sample, n_var, device=device)  # shape (n_sample, n_var)

    # Step 3: Use searchsorted to get index of selected state
    # cum_probs: (n_var, n_state) → expand to (n_sample, n_var, n_state)
    cum_probs_exp = cum_probs.unsqueeze(0).expand(n_sample, -1, -1)  # (n_sample, n_var, n_state)
    rand_vals_exp = rand_vals.unsqueeze(2)  # (n_sample, n_var, 1)

    # state_indices: (n_sample, n_var)
    state_indices = torch.sum(rand_vals_exp > cum_probs_exp, dim=2)

    # Step 4: One-hot encode
    samples = torch.nn.functional.one_hot(state_indices, num_classes=n_state).int()  # (n_sample, n_var, n_state)

    return samples


def boundary_walk(
    sfun: Callable,
    row_names: List[str],
    n_state: int,
    sys_surv_st: int,
    probs: torch.Tensor,
    rules_mat_surv: torch.Tensor,
    rules_mat_fail: torch.Tensor,
    n_walks: int = 1,
    seed: Optional[int] = None,
) -> List[Dict[str, int]]:
    """Generate boundary samples by degrading from all-operational until failure.

    Starting from the all-operational state, randomly degrade components one at
    a time (weighted by failure probability). When the system transitions from
    survival to failure, back off one step to get a "barely surviving" sample.
    Then flip the last degraded component again to get a "barely failing" sample.

    Both samples are checked against existing rules to ensure they are unknown.

    Args:
        sfun: system function (comps_st -> (fval, sys_st, info))
        row_names: component names
        n_state: max number of states per component
        sys_surv_st: system survival threshold
        probs: (n_var, n_state) probability tensor (used for degradation weights)
        rules_mat_surv: existing survival rule tensor
        rules_mat_fail: existing failure rule tensor
        n_walks: number of boundary walks to perform
        seed: random seed

    Returns:
        List of (comps_st_dict, fval, sys_st) tuples for unknown boundary samples.
    """
    rng = random.Random(seed)
    n_vars = len(row_names)
    device = probs.device

    # Build max-state (all-operational) configuration
    max_states = {}
    for i, name in enumerate(row_names):
        row = probs[i]
        nonzero = (row > 0).nonzero(as_tuple=True)[0]
        max_states[name] = int(nonzero[-1].item()) if len(nonzero) > 0 else 0

    # Build degradation weights: higher failure prob = more likely to be degraded first
    # Weight = 1 - P(best state), so components likely to fail get degraded first
    degrade_weights = []
    for i in range(n_vars):
        row = probs[i]
        nonzero = (row > 0).nonzero(as_tuple=True)[0]
        if len(nonzero) <= 1:
            degrade_weights.append(0.0)  # can't degrade single-state components
        else:
            best_p = float(row[nonzero[-1]])
            degrade_weights.append(1.0 - best_p)

    results = []

    for _ in range(n_walks):
        state = dict(max_states)
        # Components that can still be degraded
        degradable = [i for i in range(n_vars)
                      if degrade_weights[i] > 0 and state[row_names[i]] > 0]

        if not degradable:
            continue

        last_good_state = dict(state)
        last_degraded_comp = None
        n_sfun_calls = 0

        while degradable:
            # Weighted random selection
            weights = [degrade_weights[i] for i in degradable]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumsum = 0.0
            chosen_idx = degradable[0]
            for idx in degradable:
                cumsum += degrade_weights[idx]
                if cumsum >= r:
                    chosen_idx = idx
                    break

            comp_name = row_names[chosen_idx]
            prev_val = state[comp_name]
            state[comp_name] = prev_val - 1

            fval, sys_st, _ = sfun(state)
            n_sfun_calls += 1

            if sys_st < sys_surv_st:
                # System just failed — we found the boundary
                # "barely failing" sample: current state
                fail_state = dict(state)
                # "barely surviving" sample: revert last degradation
                state[comp_name] = prev_val
                surv_state = dict(state)

                # Check both against existing rules
                for candidate_state, candidate_sys_st in [(surv_state, sys_surv_st), (fail_state, sys_surv_st - 1)]:
                    # Convert to one-hot tensor for classification
                    sample_t = torch.zeros(1, n_vars, n_state, device=device)
                    for ci, name in enumerate(row_names):
                        s = candidate_state[name]
                        if s < n_state:
                            sample_t[0, ci, s] = 1.0

                    res = classify_samples(sample_t, rules_mat_surv, rules_mat_fail)
                    if res["unknown"] > 0:
                        results.append((candidate_state, fval, candidate_sys_st))

                break
            else:
                # Still surviving — record and continue degrading
                last_good_state = dict(state)
                last_degraded_comp = comp_name

                if state[comp_name] <= 0:
                    degradable.remove(chosen_idx)

        # If we exhausted all degradable components without failure,
        # the last state is a deep-survival sample (still useful if unknown)
        else:
            sample_t = torch.zeros(1, n_vars, n_state, device=device)
            for ci, name in enumerate(row_names):
                s = state[name]
                if s < n_state:
                    sample_t[0, ci, s] = 1.0
            res = classify_samples(sample_t, rules_mat_surv, rules_mat_fail)
            if res["unknown"] > 0:
                fval, sys_st, _ = sfun(state)
                results.append((state, fval, sys_st))

    return results


def make_discovery_probs(
    probs: torch.Tensor,
    bias_factor: float = 5.0,
    row_names: Optional[List[str]] = None,
    critical_components: Optional[List[str]] = None,
    critical_bias_factor: Optional[float] = None,
) -> torch.Tensor:
    """Create biased probability tensor for accelerated rule discovery.

    Shifts probability mass toward lower (degraded) states to increase the
    chance of sampling configurations in the failure/unknown regions.

    For each component, raises the failure-state probabilities by `bias_factor`
    relative to the best state, then re-normalises.  This preserves zero entries
    (padding) and keeps the tensor on the same device.

    If critical_components is provided, those components receive a higher bias
    (critical_bias_factor, default 10x bias_factor) while the rest receive the
    base bias_factor. This targets sampling toward known vulnerability clusters.

    Args:
        probs: (n_var, n_state) original probability tensor.
        bias_factor: multiplicative boost applied to non-best states.
            Higher values push more samples into degraded configurations.
            Typical range: 2-20.  A value of 1.0 returns the original probs.
        row_names: component names, required if critical_components is provided.
        critical_components: list of component names to receive extra bias.
        critical_bias_factor: bias factor for critical components.
            Default: 10 * bias_factor.

    Returns:
        discovery_probs: (n_var, n_state) biased probability tensor.
    """
    if bias_factor <= 1.0 and not critical_components:
        return probs.clone()

    dp = probs.clone()
    n_var, n_state = dp.shape

    # Build per-component bias factors
    if critical_components and row_names:
        if critical_bias_factor is None:
            critical_bias_factor = bias_factor * 10
        critical_set = set(critical_components)
        factors = [critical_bias_factor if row_names[i] in critical_set
                   else bias_factor for i in range(n_var)]
    else:
        factors = [bias_factor] * n_var

    for i in range(n_var):
        bf = factors[i]
        if bf <= 1.0:
            continue
        row = dp[i]
        # Find the best (highest-index) non-zero state
        nonzero_mask = row > 0
        if nonzero_mask.sum() <= 1:
            continue  # single-state component, nothing to bias
        best_idx = nonzero_mask.nonzero(as_tuple=True)[0][-1].item()
        # Boost all states except the best
        for s in range(n_state):
            if s != best_idx and row[s] > 0:
                dp[i, s] = row[s] * bf
        # Re-normalise
        dp[i] = dp[i] / dp[i].sum()

    return dp


def get_critical_components(
    rules: List[Dict],
    min_frequency: float = 0.3,
) -> List[str]:
    """Extract critical components from failure rules by frequency.

    Components that appear in a high fraction of failure rules are likely
    critical vulnerabilities. These can be used with make_discovery_probs()
    to create targeted biased sampling.

    Args:
        rules: list of rule dicts (from seed_rules_fail.json or rules_leq_0.json)
        min_frequency: minimum fraction of rules a component must appear in
            to be considered critical (default: 0.3 = 30%).

    Returns:
        List of component names sorted by frequency (most frequent first).
    """
    from collections import Counter
    comp_freq: Counter = Counter()
    for rule in rules:
        for k in rule:
            if k != 'sys':
                comp_freq[k] += 1
    n_rules = len(rules)
    if n_rules == 0:
        return []
    return [comp for comp, count in comp_freq.most_common()
            if count / n_rules >= min_frequency]


def fixed_k_search(
    sfun: Callable,
    row_names: List[str],
    n_state: int,
    sys_surv_st: int,
    probs: torch.Tensor,
    k: int = 3,
    n_samples: int = 100_000,
    n_workers: int = 1,
    priority_components: Optional[List[str]] = None,
    worst_state: bool = True,
    seed: Optional[int] = None,
) -> List[Tuple[Dict[str, int], float, int]]:
    """Search for failure modes by randomly sampling k degraded components.

    Randomly selects k components to degrade from their best state, weighted
    by their failure probability, while keeping all other components fully
    operational. For each sample, calls sfun to check if the system fails.

    Args:
        sfun: system function (comps_st -> (fval, sys_st, info))
        row_names: component names
        n_state: max number of states per component
        sys_surv_st: system survival state threshold
        probs: (n_var, n_state) probability tensor
        k: number of components to degrade simultaneously
        n_samples: number of random k-combinations to test
        n_workers: number of parallel workers (uses multiprocessing)
        priority_components: if set, at least one component in each combo
            must be from this list
        worst_state: if True, degrade to state 0 (worst); if False, sample
            degraded state weighted by probability
        seed: random seed

    Returns:
        List of (comps_st_dict, fval, sys_st) for each failure found.
    """
    rng = random.Random(seed)
    n_vars = len(row_names)

    # Build max-state (all-operational) configuration
    max_states = {}
    degradable_states = {}  # component -> list of degraded states
    degrade_weights = {}  # component -> weight (1 - p_best)
    for i, name in enumerate(row_names):
        row = probs[i]
        nonzero = (row > 0).nonzero(as_tuple=True)[0]
        if len(nonzero) > 0:
            best = int(nonzero[-1].item())
            max_states[name] = best
            deg = [int(s.item()) for s in nonzero if int(s.item()) < best]
            if deg:
                degradable_states[name] = deg
                degrade_weights[name] = 1.0 - float(row[best].item())
        else:
            max_states[name] = 0

    degradable_names = list(degradable_states.keys())
    weights = [degrade_weights[n] for n in degradable_names]
    total_weight = sum(weights)

    mode = "worst-state" if worst_state else "prob-weighted"
    print(f"Fixed-k search: k={k}, {len(degradable_names)} degradable components, "
          f"{n_samples} samples, {mode}")

    # Build priority index if specified
    priority_indices = None
    if priority_components:
        priority_set = set(priority_components) & set(degradable_names)
        priority_indices = [i for i, n in enumerate(degradable_names)
                            if n in priority_set]
        pri_weights = [weights[i] for i in priority_indices]
        print(f"  Priority: {len(priority_indices)} components")

    # Generate random k-combinations weighted by failure probability
    tasks = []
    seen = set()
    attempts = 0
    max_attempts = n_samples * 10

    while len(tasks) < n_samples and attempts < max_attempts:
        attempts += 1

        if priority_indices and rng.random() < 0.5:
            # Force at least one priority component
            n_pri = rng.randint(1, min(k, len(priority_indices)))
            pri_chosen = _weighted_sample_without_replacement(
                rng, priority_indices, pri_weights, n_pri)
            remaining_indices = [i for i in range(len(degradable_names))
                                 if i not in set(pri_chosen)]
            remaining_weights = [weights[i] for i in remaining_indices]
            n_other = k - n_pri
            if n_other > len(remaining_indices):
                continue
            other_chosen = _weighted_sample_without_replacement(
                rng, remaining_indices, remaining_weights, n_other)
            chosen = sorted(pri_chosen + other_chosen)
        else:
            chosen = sorted(_weighted_sample_without_replacement(
                rng, list(range(len(degradable_names))), weights, k))

        key = tuple(chosen)
        if key in seen:
            continue
        seen.add(key)

        # Build state: all operational except chosen components
        state = dict(max_states)
        for idx in chosen:
            name = degradable_names[idx]
            deg_states = degradable_states[name]
            if worst_state:
                state[name] = min(deg_states)  # state 0 (worst)
            elif len(deg_states) == 1:
                state[name] = deg_states[0]
            else:
                # Weight by actual probabilities of degraded states
                deg_probs = [float(probs[row_names.index(name), s].item())
                             for s in deg_states]
                state[name] = _weighted_choice(rng, deg_states, deg_probs)

        tasks.append(state)

    print(f"  Generated {len(tasks)} unique combinations "
          f"({attempts} attempts)")

    # Execute
    failures = []
    n_tested = 0

    if n_workers > 1:
        global _MP_SFUN, _MP_SYS_SURV_ST, _MP_N_STATE
        _MP_SFUN = sfun
        _MP_SYS_SURV_ST = sys_surv_st
        _MP_N_STATE = n_state

        batch_size = max(n_workers * 4, 100)
        with mp.Pool(n_workers) as pool:
            for batch_start in range(0, len(tasks), batch_size):
                batch = tasks[batch_start:batch_start + batch_size]
                results = pool.map(_eval_sfun_worker, batch)
                for comps_st, fval, sys_st in results:
                    n_tested += 1
                    if sys_st < sys_surv_st:
                        failures.append((comps_st, fval, sys_st))
                if n_tested % 10000 == 0 or batch_start + batch_size >= len(tasks):
                    print(f"  Tested {n_tested}/{len(tasks)}, "
                          f"failures found: {len(failures)}", flush=True)
    else:
        for comps_st in tasks:
            fval, sys_st, _ = sfun(comps_st)
            n_tested += 1
            if sys_st < sys_surv_st:
                failures.append((comps_st, fval, sys_st))
            if n_tested % 10000 == 0 or n_tested == len(tasks):
                print(f"  Tested {n_tested}/{len(tasks)}, "
                      f"failures found: {len(failures)}", flush=True)

    print(f"Fixed-k search complete: {len(failures)} failures in "
          f"{n_tested} evaluations")
    return failures


def fixed_k_survival_search(
    sfun: Callable,
    row_names: List[str],
    n_state: int,
    sys_surv_st: int,
    probs: torch.Tensor,
    k: int = 10,
    n_samples: int = 100_000,
    n_workers: int = 1,
    priority_components: Optional[List[str]] = None,
    best_state: bool = True,
    seed: Optional[int] = None,
    target_components: Optional[List[str]] = None,
) -> List[Tuple[Dict[str, int], float, int]]:
    """Search for survival modes by keeping k components operational, rest degraded.

    Inverse of fixed_k_search: randomly selects k components to keep at their
    best state while degrading the rest to worst state. For each sample,
    calls sfun to check if the system survives.

    Args:
        sfun: system function (comps_st -> (fval, sys_st, info))
        row_names: component names
        n_state: max number of states per component
        sys_surv_st: system survival state threshold
        probs: (n_var, n_state) probability tensor
        k: number of target components to keep operational
        n_samples: number of random k-combinations to test
        n_workers: number of parallel workers (uses multiprocessing)
        priority_components: if set, at least one component in each combo
            must be from this list
        best_state: if True, keep selected at best state (default);
            if False, sample state weighted by probability
        seed: random seed
        target_components: if set, only these components are candidates for
            keeping/degrading. All other components stay at best state.
            This restricts the search to a subset (e.g., only generators).

    Returns:
        List of (comps_st_dict, fval, sys_st) for each survival found.
    """
    rng = random.Random(seed)

    target_set = set(target_components) if target_components else None

    # Build max-state and worst-state configurations
    max_states = {}
    min_states = {}
    degradable_states = {}
    keep_weights = {}
    for i, name in enumerate(row_names):
        row = probs[i]
        nonzero = (row > 0).nonzero(as_tuple=True)[0]
        if len(nonzero) > 0:
            best = int(nonzero[-1].item())
            worst = int(nonzero[0].item())
            max_states[name] = best
            min_states[name] = worst
            # Only consider target components for degradation
            if best > worst and (target_set is None or name in target_set):
                degradable_states[name] = True
                keep_weights[name] = float(row[best].item())
            else:
                keep_weights[name] = 0.0
        else:
            max_states[name] = 0
            min_states[name] = 0

    keepable_names = [n for n in row_names if n in degradable_states]
    weights = [keep_weights[n] for n in keepable_names]

    mode = "best-state" if best_state else "prob-weighted"
    print(f"Fixed-k survival search: k={k}, {len(keepable_names)} components, "
          f"{n_samples} samples, {mode}")

    # Build priority index if specified
    priority_indices = None
    if priority_components:
        priority_set = set(priority_components) & set(keepable_names)
        priority_indices = [i for i, n in enumerate(keepable_names)
                            if n in priority_set]
        pri_weights = [weights[i] for i in priority_indices]
        print(f"  Priority: {len(priority_indices)} components")

    # Generate random k-combinations
    tasks = []
    seen = set()
    attempts = 0
    max_attempts = n_samples * 10

    while len(tasks) < n_samples and attempts < max_attempts:
        attempts += 1

        if priority_indices and rng.random() < 0.5:
            n_pri = rng.randint(1, min(k, len(priority_indices)))
            pri_chosen = _weighted_sample_without_replacement(
                rng, priority_indices, pri_weights, n_pri)
            remaining_indices = [i for i in range(len(keepable_names))
                                 if i not in set(pri_chosen)]
            remaining_weights = [weights[i] for i in remaining_indices]
            n_other = k - n_pri
            if n_other > len(remaining_indices):
                continue
            other_chosen = _weighted_sample_without_replacement(
                rng, remaining_indices, remaining_weights, n_other)
            chosen = sorted(pri_chosen + other_chosen)
        else:
            chosen = sorted(_weighted_sample_without_replacement(
                rng, list(range(len(keepable_names))), weights, k))

        key = tuple(chosen)
        if key in seen:
            continue
        seen.add(key)

        # Build state: target components degraded except chosen (kept operational),
        # non-target components stay at best state
        if target_set is not None:
            # Start with all at best, then degrade only target components
            state = dict(max_states)
            for name in keepable_names:
                state[name] = min_states[name]
        else:
            state = dict(min_states)
        for idx in chosen:
            name = keepable_names[idx]
            state[name] = max_states[name]

        tasks.append(state)

    print(f"  Generated {len(tasks)} unique combinations "
          f"({attempts} attempts)")

    # Execute
    survivals = []
    n_tested = 0

    if n_workers > 1:
        global _MP_SFUN, _MP_SYS_SURV_ST, _MP_N_STATE
        _MP_SFUN = sfun
        _MP_SYS_SURV_ST = sys_surv_st
        _MP_N_STATE = n_state

        batch_size = max(n_workers * 4, 100)
        with mp.Pool(n_workers) as pool:
            for batch_start in range(0, len(tasks), batch_size):
                batch = tasks[batch_start:batch_start + batch_size]
                results = pool.map(_eval_sfun_worker, batch)
                for comps_st, fval, sys_st in results:
                    n_tested += 1
                    if sys_st >= sys_surv_st:
                        survivals.append((comps_st, fval, sys_st))
                if n_tested % 10000 == 0 or batch_start + batch_size >= len(tasks):
                    print(f"  Tested {n_tested}/{len(tasks)}, "
                          f"survivals found: {len(survivals)}", flush=True)
    else:
        for comps_st in tasks:
            fval, sys_st, _ = sfun(comps_st)
            n_tested += 1
            if sys_st >= sys_surv_st:
                survivals.append((comps_st, fval, sys_st))
            if n_tested % 10000 == 0 or n_tested == len(tasks):
                print(f"  Tested {n_tested}/{len(tasks)}, "
                      f"survivals found: {len(survivals)}", flush=True)

    print(f"Fixed-k survival search complete: {len(survivals)} survivals in "
          f"{n_tested} evaluations")
    return survivals


def _weighted_sample_without_replacement(rng, indices, weights, k):
    """Sample k items from indices without replacement, weighted."""
    if k >= len(indices):
        return list(indices)
    selected = []
    available = list(indices)
    avail_weights = list(weights)
    for _ in range(k):
        total = sum(avail_weights)
        r = rng.random() * total
        cumsum = 0.0
        for j, (idx, w) in enumerate(zip(available, avail_weights)):
            cumsum += w
            if cumsum >= r:
                selected.append(idx)
                available.pop(j)
                avail_weights.pop(j)
                break
    return selected


def _weighted_choice(rng, items, weights):
    """Weighted random choice from items."""
    total = sum(weights)
    r = rng.random() * total
    cumsum = 0.0
    for item, w in zip(items, weights):
        cumsum += w
        if cumsum >= r:
            return item
    return items[-1]


def _eval_sfun_worker(comps_st):
    """Worker for parallel sfun evaluation in fixed_k_search."""
    sfun = _MP_SFUN
    sys_surv_st = _MP_SYS_SURV_ST
    fval, sys_st, _ = sfun(comps_st)
    return comps_st, fval, sys_st


def mask_from_first_one(
    x: torch.Tensor,
    mode: str = "after"
) -> torch.Tensor:
    """
    Create masks relative to the first 1 in each row.

    Args:
        x: (n_row, n_col) or (batch, n_row, n_col) int/bool tensor with 0/1 entries
        mode:
            - "after"  → ones from first 1 (inclusive) to end
            - "before" → ones from start up to first 1 (inclusive)
    Returns:
        Tensor of same shape as x, dtype=int32, device preserved.
    """
    assert x.ndim in (2, 3), "x must be 2D or 3D"
    device = x.device

    # Normalize to 3D: (B, N, M)
    squeeze_back = (x.ndim == 2)
    if squeeze_back:
        x3 = x.unsqueeze(0)
    else:
        x3 = x

    B, N, M = x3.shape

    # Column indices for broadcasting comparisons
    cols = torch.arange(M, device=device).view(1, 1, M).expand(B, N, M)

    # First index of "1" per row
    x_bool = (x3 == 1) if x3.dtype != torch.bool else x3
    has_one = x_bool.any(dim=2)                 # (B, N)
    first_idx = x_bool.int().argmax(dim=2)      # (B, N); 0 if none
    first_idx = torch.where(has_one, first_idx, torch.full_like(first_idx, M))

    if mode == "after":
        mask = cols >= first_idx.unsqueeze(-1)  # (B, N, M)
    elif mode == "before":
        mask = cols <= first_idx.unsqueeze(-1)  # (B, N, M)
    else:
        raise ValueError("mode must be 'after' or 'before'")

    mask = mask.to(torch.int32)

    return mask.squeeze(0) if squeeze_back else mask

def update_rules(min_comps_st, rules_dict, rules_mat, row_names, verbose=False):
    _, _, n_state = rules_mat.shape
    Rnew = from_rule_dict_to_mat(min_comps_st, row_names, n_state)
    is_Rnew_subset, are_Rset_subset = is_subset(Rnew, rules_mat)

    if is_Rnew_subset:
        if verbose:
            print("WARNING: New rule is a subset of existing rules. No update made.")
        return rules_dict, rules_mat

    rules_mat = rules_mat[~are_Rset_subset,:,:]
    rules_dict = [r for r, keep in zip(rules_dict, ~are_Rset_subset) if keep]

    rules_dict.append(min_comps_st)
    rules_mat = torch.cat((rules_mat, Rnew.unsqueeze(0)), dim=0)
    if verbose:
        print("No. of existing rules removed: ", int(sum(are_Rset_subset)))

    return rules_dict, rules_mat


def update_rules_batch(new_rules_dicts, rules_dict, rules_mat, row_names, verbose=False):
    """
    Batch version of update_rules: process multiple new rules at once.

    Instead of calling is_subset N times (each against a growing rules_mat),
    this does:
      1. Convert all new rules to matrices in one pass
      2. One batched dominance check: new vs existing
      3. One batched dominance check: new vs new (inter-batch)
      4. Filter and append all surviving rules at once

    Returns:
        (rules_dict, rules_mat, n_added, n_removed)
    """
    if not new_rules_dicts:
        return rules_dict, rules_mat, 0, 0

    n_existing, n_var, n_state = rules_mat.shape
    device = rules_mat.device

    # Step 1: convert all new rules to matrices
    new_mats = []
    for rd in new_rules_dicts:
        new_mats.append(from_rule_dict_to_mat(rd, row_names, n_state))
    new_batch = torch.stack(new_mats, dim=0)  # (N_new, n_var, n_state)
    n_new = new_batch.shape[0]

    # Step 2: check new vs existing
    # For each new rule, is it dominated by any existing rule?
    # For each existing rule, is it dominated by any new rule?
    # new_batch: (N_new, n_var, n_state), rules_mat: (N_ex, n_var, n_state)
    new_dominated = torch.zeros(n_new, dtype=torch.bool, device=device)
    existing_dominated = torch.zeros(n_existing, dtype=torch.bool, device=device)

    if n_existing > 0 and n_new > 0:
        # Chunk over existing rules to bound memory: (N_new, chunk, n_var, n_state)
        # With N_new=96, chunk=8000, n_var=120, n_state=2: ~180MB per chunk
        chunk_size = max(1, 500_000_000 // (n_new * n_var * n_state * 4))  # ~500MB limit
        for c_start in range(0, n_existing, chunk_size):
            c_end = min(c_start + chunk_size, n_existing)
            ex_chunk = rules_mat[c_start:c_end]  # (chunk, n_var, n_state)
            new_exp = new_batch.unsqueeze(1)      # (N_new, 1, n_var, n_state)
            ex_exp = ex_chunk.unsqueeze(0)        # (1, chunk, n_var, n_state)
            intersect = new_exp & ex_exp          # (N_new, chunk, n_var, n_state)

            # new[i] dominated by existing[j]?
            new_eq = (new_exp == intersect).all(dim=(2, 3))  # (N_new, chunk)
            new_dominated |= new_eq.any(dim=1)

            # existing[j] dominated by new[i]?
            ex_eq = (ex_exp == intersect).all(dim=(2, 3))    # (N_new, chunk)
            existing_dominated[c_start:c_end] |= ex_eq.any(dim=0)

    # Step 3: among surviving new rules, check inter-dominance
    surviving_new_idx = torch.where(~new_dominated)[0]
    if len(surviving_new_idx) > 1:
        surv_batch = new_batch[surviving_new_idx]  # (M, n_var, n_state)
        M = surv_batch.shape[0]
        s_exp_i = surv_batch.unsqueeze(1)  # (M, 1, n_var, n_state)
        s_exp_j = surv_batch.unsqueeze(0)  # (1, M, n_var, n_state)
        s_inter = s_exp_i & s_exp_j        # (M, M, n_var, n_state)
        # i is subset of j: s_exp_i == s_inter
        i_sub_j = (s_exp_i == s_inter).all(dim=(2, 3))  # (M, M)
        j_sub_i = (s_exp_j == s_inter).all(dim=(2, 3))  # (M, M)
        # Mask diagonal
        i_sub_j.fill_diagonal_(False)
        j_sub_i.fill_diagonal_(False)
        # Strict dominance: j dominates i (i⊂j but j⊄i)
        strict = i_sub_j & ~j_sub_i
        # Equal rules (i⊂j AND j⊂i): tiebreak by index — only j<i can dominate i
        equal = i_sub_j & j_sub_i
        lower_mask = torch.tril(torch.ones(M, M, dtype=torch.bool, device=device), diagonal=-1)
        # Rule i is dominated if strictly dominated by any j, or equal to some j<i
        inter_dominated = strict.any(dim=1) | (equal & lower_mask).any(dim=1)  # (M,)
        # Map back: mark dominated ones
        dominated_in_surviving = surviving_new_idx[inter_dominated]
        new_dominated[dominated_in_surviving] = True

    # Step 4: filter existing, append surviving new
    keep_existing = ~existing_dominated
    keep_new = ~new_dominated

    n_removed = int(existing_dominated.sum().item())
    n_added = int(keep_new.sum().item())

    rules_mat = torch.cat([
        rules_mat[keep_existing],
        new_batch[keep_new],
    ], dim=0)

    rules_dict = [r for r, k in zip(rules_dict, keep_existing.tolist()) if k]
    for i, rd in enumerate(new_rules_dicts):
        if keep_new[i]:
            rules_dict.append(rd)

    if verbose:
        print(f"Batch update: {n_added} rules added, {n_removed} existing rules removed "
              f"({n_new - n_added} new rules dominated)")

    return rules_dict, rules_mat, n_added, n_removed

def run_rule_extraction(
    *,
    # Problem-specific callables / data
    sfun: Callable[[Dict[str, int]], Tuple[Any, Any, Any]],
    probs: Tensor,
    row_names: List[str],
    n_state: int,
    rules_surv: List[Dict[str, Any]] = [],
    rules_fail: List[Dict[str, Any]] = [],
    rules_mat_surv: Tensor = None,
    rules_mat_fail: Tensor = None,
    # Analysis parameters
    stochastic_search: bool = True,
    gamma: float = 0.5, # if stochastic_search==False, ignored. 0 < γ < 1 → more emphasis on exploration; γ > 1 → more emphasis on exploitation
    # Termination / threshold settings
    unk_prob_thres: float = 5e-2,
    # Frequencies / sampling settings
    prob_update_every: int = 50,   # (2) how often to test system probabilities/bounds
    save_every: int = 10,          # (4) how often to persist logs/rules
    n_sample: int = 1_000_000,
    sample_batch_size: int = 1_000_000,
    rule_search_batch_size: int = 1_024,    # sampler batch for candidate rule search
    rule_search_max_iters: int = 10,
    min_rule_search: bool = True, # May be opted out for expensive sfun
    # Display / verbose
    rule_update_verbose: bool = True,
    # Output control
    output_dir: str = "tsum_temp",
    surv_json_name: str = "rules_surv.json",
    fail_json_name: str = "rules_fail.json",
    surv_pt_name: str = "rules_surv.pt",
    fail_pt_name: str = "rules_fail.pt",
    metrics_path: str = "metrics.jsonl",
) -> Dict[str, Any]:
    """
    Runs the survival/failure rule discovery loop (steps 3 & 4 only),
    periodically evaluates unknown probability via sampling, and logs metrics.

    Returns a dict with updated rules, rule matrices, threshold lists, and the in-memory metrics log.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ---- helpers ----
    def _avg_rule_len(rule_store: Any) -> float:
        """
        Try to estimate average number of conditions in current rules.
        Length of rule dictionary minus system event: len(rule) - 1
        Works for list-of-dictionaries; returns 0.0 if unavailable.
        """
        try:
            if rule_store is None:
                return 0.0
            # If it's a list-like of rules:
            if hasattr(rule_store, "__len__") and len(rule_store) > 0:
                total = sum([len(r) - 1 for r in rule_store])
                count = len(rule_store)
                return float(total) / count
        except Exception:
            pass
        return 0.0

    def _save_json(obj, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)

    def _save_pt(t: torch.Tensor, path: str) -> None:
        torch.save(t.detach().cpu(), path)

    # ---- initial state ----
    device = probs.device
    n_sample_loop = max(int(n_sample // sample_batch_size), 1)

    unk_prob = 1.0
    n_round = 0
    metrics_log: List[Dict[str, Any]] = []

    n_vars = len(row_names)
    if rules_mat_surv is None:
        rules_mat_surv = torch.empty((0,n_vars,n_state), dtype=torch.int32, device=device)
    if rules_mat_fail is None:
        rules_mat_fail = torch.empty((0,n_vars,n_state), dtype=torch.int32, device=device)

    # Threshold discovery bookkeeping
    sys_val_list = []

    # JSONL file for metrics (append-only)
    metrics_path = os.path.join(output_dir, metrics_path)
    # snapshot rules paths
    rules_surv_path = os.path.join(output_dir, surv_json_name)
    rules_fail_path = os.path.join(output_dir, fail_json_name)
    rules_surv_pt_path = os.path.join(output_dir, surv_pt_name)
    rules_fail_pt_path = os.path.join(output_dir, fail_pt_name)

    # while flags: use only steps 3 & 4 flags
    is_new_surv_cand, is_new_fail_cand = True, True

    # last known probabilities (only updated when recomputed)
    last_probs = {"survival": None, "failure": None, "unknown": None}
    

    # ---- main loop ----
    while (is_new_surv_cand or is_new_fail_cand) and (unk_prob > unk_prob_thres):
        n_round += 1
        t0 = time.perf_counter()

        print("---")
        print(f"Round: {n_round}, Unk. prob.: {unk_prob:.3e}")
        print(f"No. of non-dominant rules: {len(rules_mat_surv)+len(rules_mat_fail)}, "
              f"Survival rules: {len(rules_mat_surv)}, Failure rules: {len(rules_mat_fail)}")

        # ---- 3) Get a survival candidate from survival rules ----
        is_new_surv_cand, rules_surv, rules_fail, rules_mat_surv, rules_mat_fail, sys_val_list = \
            run_survival_candidate_round(
                probs=probs,
                rules_mat_surv=rules_mat_surv,
                rules_mat_fail=rules_mat_fail,
                rules_surv=rules_surv,
                rules_fail=rules_fail,
                row_names=row_names,
                n_state=n_state,
                sys_val_list=sys_val_list,
                sfun=sfun,
                rule_search_batch_size=rule_search_batch_size,
                rule_search_max_iters=rule_search_max_iters,
                stochastic_search=stochastic_search,
                gamma=gamma,
                min_rule_search=min_rule_search,
                rule_update_verbose=rule_update_verbose,
            )

        # ---- 4) Get a failure candidate from failure rules ----
        is_new_fail_cand, rules_surv, rules_fail, rules_mat_surv, rules_mat_fail, sys_val_list = \
            run_failure_candidate_round(
                probs=probs,
                rules_mat_surv=rules_mat_surv,
                rules_mat_fail=rules_mat_fail,
                rules_surv=rules_surv,
                rules_fail=rules_fail,
                row_names=row_names,
                n_state=n_state,
                sys_val_list=sys_val_list,
                sfun=sfun,
                rule_search_batch_size=rule_search_batch_size,
                rule_search_max_iters=rule_search_max_iters,
                stochastic_search=stochastic_search,
                gamma=gamma,
                min_rule_search=min_rule_search,
                rule_update_verbose=rule_update_verbose,
            )

        # ---- Periodic probability (bound) test via sampling ----
        probs_updated = False
        if (n_round % prob_update_every) == 0:
            total_loops = max(n_sample // sample_batch_size, 1)
            counts = {"survival": 0, "failure": 0, "unknown": 0}
            for i in range(total_loops):
                samples = sample_categorical(probs, sample_batch_size)
                counts_i = classify_samples(samples, rules_mat_surv, rules_mat_fail)
                counts["survival"] += counts_i["survival"]
                counts["failure"] += counts_i["failure"]
                counts["unknown"] += counts_i["unknown"]

            samp_probs = {k: v / (sample_batch_size * total_loops) for k, v in counts.items()}
            print("---")
            print(f"Probs: 'surv': {samp_probs['survival']: .3e}, 'fail': {samp_probs['failure']: .3e}, 'unkn': {samp_probs['unknown']: .3e}")
            unk_prob = samp_probs["unknown"]
            last_probs.update(samp_probs)
            probs_updated = True

        # ---- metrics for this round ----
        dt = time.perf_counter() - t0
        entry = {
            "round": n_round,
            "time_sec": dt,
            "n_rules_surv": int(len(rules_mat_surv)),
            "n_rules_fail": int(len(rules_mat_fail)),
            "probs_updated": probs_updated,
            "p_survival": last_probs["survival"] if probs_updated else None,
            "p_failure": last_probs["failure"] if probs_updated else None,
            "p_unknown": last_probs["unknown"] if probs_updated else None,
            "avg_len_surv": _avg_rule_len(rules_surv),
            "avg_len_fail": _avg_rule_len(rules_fail),
        }
        metrics_log.append(entry)

        # ---- periodic persistence of metrics and rules ----
        if (n_round % save_every) == 0:
            # append metrics as JSONL
            with open(metrics_path, "a", encoding="utf-8") as mf:
                for e in metrics_log[-save_every:]:
                    mf.write(json.dumps(e) + "\n")
            # snapshot rules
            _save_json(rules_surv, rules_surv_path)
            _save_json(rules_fail, rules_fail_path)
            _save_pt(rules_mat_surv, rules_surv_pt_path)
            _save_pt(rules_mat_fail, rules_fail_pt_path)

    # Final flush of any remaining metrics not yet written by save_every
    last_flushed_rounds = (n_round // save_every) * save_every
    if last_flushed_rounds < n_round and metrics_log:
        with open(metrics_path, "a", encoding="utf-8") as mf:
            for e in metrics_log[last_flushed_rounds:]:
                mf.write(json.dumps(e) + "\n")
    # Final snapshot of rules
    _save_json(rules_surv, rules_surv_path)
    _save_json(rules_fail, rules_fail_path)
    _save_pt(rules_mat_surv, rules_surv_pt_path)
    _save_pt(rules_mat_fail, rules_fail_pt_path)
    # Final probability check
    total_loops = max(n_sample // sample_batch_size, 1)
    counts = {"survival": 0, "failure": 0, "unknown": 0}
    for i in range(total_loops):
        samples = sample_categorical(probs, sample_batch_size)
        counts_i = classify_samples(samples, rules_mat_surv, rules_mat_fail)
        counts["survival"] += counts_i["survival"]
        counts["failure"] += counts_i["failure"]
        counts["unknown"] += counts_i["unknown"]

    samp_probs = {k: v / (sample_batch_size * total_loops) for k, v in counts.items()}
    print("---")
    print(f"[Final results] Probs: 'surv': {samp_probs['survival']: .3e}, 'fail': {samp_probs['failure']: .3e}, 'unkn': {samp_probs['unknown']: .3e}")
    unk_prob = samp_probs["unknown"]
    last_probs.update(samp_probs)
    probs_updated = True
    # ---
    dt = time.perf_counter() - t0
    entry = {
        "round": n_round,
        "time_sec": dt,
        "n_rules_surv": int(len(rules_mat_surv)),
        "n_rules_fail": int(len(rules_mat_fail)),
        "probs_updated": probs_updated,
        "p_survival": last_probs["survival"] if probs_updated else None,
        "p_failure": last_probs["failure"] if probs_updated else None,
        "p_unknown": last_probs["unknown"] if probs_updated else None,
        "avg_len_surv": _avg_rule_len(rules_surv),
        "avg_len_fail": _avg_rule_len(rules_fail),
    }
    metrics_log.append(entry)

    return {
        "sys_vals": sys_val_list,
        "metrics_path": metrics_path,
        "rules_surv_path": rules_surv_path,
        "rules_fail_path": rules_fail_path,
        "rules_surv_pt_path": rules_surv_pt_path,   
        "rules_fail_pt_path": rules_fail_pt_path,
        "metrics_log": metrics_log,  # also returned in-memory
    }

def mixed_sort_key(x):
    if x is None:
        return (2, 0, 0.0, "")
    is_numeric = (
        isinstance(x, (int, float, Decimal)) and not isinstance(x, bool)
    ) or isinstance(x, _NUMPY_NUM)
    if is_numeric:
        v = float(x)
        if math.isnan(v):
            return (0, 1, 0.0, "")
        return (0, 0, v, "")
    if isinstance(x, str):
        return (1, 0, 0.0, x.lower())
    return (1, 0, 0.0, str(x).lower())

def classify_samples_with_indices(
    samples: torch.Tensor,
    survival_rules: List[torch.Tensor],
    failure_rules: List[torch.Tensor],
    *,
    return_masks: bool = False
) -> Dict[str, Any]:
    """
    Classify samples as survival, failure, or unknown using subset checks,
    and return indices for each class.

    Args:
        samples: (n_sample, n_var, n_state) binary tensor
        survival_rules: list of rule tensors, each (n_var, n_state) or (n_var+1, n_state)
        failure_rules: list of rule tensors, each (n_var, n_state) or (n_var+1, n_state)
        return_masks: if True, also return boolean masks per class

    Returns:
        {
          'survival': int,
          'failure' : int,
          'unknown' : int,
          'idx_survival': LongTensor[ns],
          'idx_failure' : LongTensor[nf],
          'idx_unknown' : LongTensor[nu],
          # optionally:
          'mask_survival': BoolTensor[n_sample],
          'mask_failure' : BoolTensor[n_sample],
          'mask_unknown' : BoolTensor[n_sample],
        }
    """
    device = samples.device
    n_sample = samples.shape[0]
    survival_rules = _ensure_rules_tensor(survival_rules, device)
    failure_rules = _ensure_rules_tensor(failure_rules, device)

    samples_flat = samples.reshape(n_sample, -1).to(dtype=torch.float16)

    # Survival check
    survival_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)
    if survival_rules.ndim == 3 and survival_rules.shape[0] > 0:
        not_surv = (~survival_rules.bool()).reshape(
            survival_rules.shape[0], -1).to(dtype=torch.float16)
        survival_mask = _check_any_subset(samples_flat, not_surv)

    # Failure check (only on non-survival samples)
    failure_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)
    remaining = ~survival_mask
    if failure_rules.ndim == 3 and failure_rules.shape[0] > 0 and remaining.any():
        not_fail = (~failure_rules.bool()).reshape(
            failure_rules.shape[0], -1).to(dtype=torch.float16)
        fail_sub = _check_any_subset(samples_flat[remaining], not_fail)
        failure_mask[remaining] = fail_sub

    unknown_mask = ~survival_mask & ~failure_mask

    # Indices
    idx_survival = torch.where(survival_mask)[0]
    idx_failure  = torch.where(failure_mask)[0]
    idx_unknown  = torch.where(unknown_mask)[0]

    result: Dict[str, Any] = {
        'survival': int(survival_mask.sum().item()),
        'failure' : int(failure_mask.sum().item()),
        'unknown' : int(unknown_mask.sum().item()),
        'idx_survival': idx_survival,
        'idx_failure' : idx_failure,
        'idx_unknown' : idx_unknown,
    }

    if return_masks:
        result['mask_survival'] = survival_mask
        result['mask_failure']  = failure_mask
        result['mask_unknown']  = unknown_mask

    return result

def get_comp_cond_sys_prob(
    rules_mat_surv: Tensor,
    rules_mat_fail: Tensor,
    probs: Tensor,
    comps_st_cond: Dict[str, int],
    row_names: Sequence[str],
    s_fun: callable = None,                          # Callable[[Dict[str,int]], tuple]
    sys_surv_st: int = 1,        # system state value indicating survival
    n_sample: int = 1_000_000,
    n_batch:  int = 1_000_000
) -> Dict[str, float]:
    """
    P(system state | given component states).

    - 'probs' is (n_var, n_state) categorical; we condition rows listed in comps_st_cond to one-hot.
    - We classify samples using rules; for unknowns we call s_fun(comps_dict) to resolve.
    - Returns probabilities over {'survival','failure'} that sum ~ 1.0.

    """
    # --- clone probs and apply conditioning ---
    if torch.is_tensor(probs):
        probs_cond = probs.clone()
        n_comps, n_states = probs_cond.shape
    else:
        raise TypeError("Expected 'probs' to be a torch.Tensor of shape (n_var, n_state).")

    if len(row_names) != n_comps:
        raise ValueError(f"row_names length ({len(row_names)}) must match probs rows ({n_comps}).")

    for x, s in comps_st_cond.items():
        try:
            row_idx = row_names.index(x)
        except ValueError:
            raise ValueError(f"Component {x} not found in row_names.")
        if not (0 <= int(s) < n_states):
            raise ValueError(f"State {s} for component {x} is out of bounds [0,{n_states-1}].")
        probs_cond[row_idx].zero_()
        probs_cond[row_idx, int(s)] = 1.0

    # --- sampling loop (exactly n_sample draws) ---
    batch_size = max(1, min(int(n_batch), int(n_sample)))
    remaining = int(n_sample)

    counts = {"survival": 0, "failure": 0, "unknown": 0}

    while remaining > 0:
        b = min(batch_size, remaining)
        # IMPORTANT: sample from the *conditioned* probs
        samples = sample_categorical(probs_cond, b)  # (b, n_var, n_state) one-hot

        res = classify_samples_with_indices(
            samples, rules_mat_surv, rules_mat_fail, return_masks=True
        )

        counts["survival"] += int(res["survival"])
        counts["failure"]  += int(res["failure"])

        idx_unknown = res["idx_unknown"]
        if idx_unknown.numel() > 0:
            if s_fun is not None:
                # Resolve unknowns with s_fun
                for j in idx_unknown.tolist():
                    sample_j = samples[j]  # (n_var, n_state)
                    # convert one-hot row -> state index per var
                    states = torch.argmax(sample_j, dim=1).tolist()

                    # build comps dict for s_fun
                    comps = {row_names[k]: int(states[k]) for k in range(n_comps)}

                    _, sys_st, _ = s_fun(comps)

                    if sys_st >= sys_surv_st:
                        counts["survival"] += 1
                    else:
                        counts["failure"] += 1

            else:
                counts["unknown"] += int(idx_unknown.shape[0])

        remaining -= b

    # --- normalize to probabilities (denominator = requested n_sample) ---
    total = float(n_sample)
    cond_probs = {k: counts[k] / total for k in counts}
    return cond_probs

def get_comp_cond_sys_prob_multi(
    rules_dict_surv: Dict[int, Tensor],
    rules_dict_fail: Dict[int, Tensor],
    probs: Tensor,
    comps_st_cond: Dict[str, int],
    row_names: Sequence[str],
    s_fun: callable = None,                          # Callable[[Dict[str,int]], tuple]
    n_sample: int = 1_000_000,
    n_batch:  int = 1_000_000
) -> Dict[str, float]:
    """
    Estimate P(system state = s | given component states) for multi-state systems by Monte Carlo.

    Args:
        rules_dict_surv: dict of system survival rule tensors {state: Tensor(n_var, n_state)}.
        rules_dict_fail: dict of system failure rule tensors {state: Tensor(n_var, n_state)}.
        probs: (n_var, n_state) categorical probability tensor.
        comps_st_cond: dict of known component states {name: state_index}.
        row_names: list of variable (component) names matching probs rows.
        s_fun: function(comps_dict) -> tuple(_, sys_state, _).
        n_sample, n_batch: number of samples total and per batch.

    Returns:
        Dictionary {state: probability}, summing to 1.0.
    """
    # --- clone probs and apply conditioning ---
    if torch.is_tensor(probs):
        probs_cond = probs.clone()
        n_comps, n_states = probs_cond.shape
    else:
        raise TypeError("Expected 'probs' to be a torch.Tensor of shape (n_var, n_state).")

    if len(row_names) != n_comps:
        raise ValueError(f"row_names length ({len(row_names)}) must match probs rows ({n_comps}).")

    # Applying conditioning
    for x, s in comps_st_cond.items():
        try:
            row_idx = row_names.index(x)
        except ValueError:
            raise ValueError(f"Component {x} not found in row_names.")
        if not (0 <= int(s) < n_states):
            raise ValueError(f"State {s} for component {x} is out of bounds [0,{n_states-1}].")
        probs_cond[row_idx].zero_()
        probs_cond[row_idx, int(s)] = 1.0

    # Validate rule keys
    keys_surv = set(rules_dict_surv.keys())
    keys_fail = set(rules_dict_fail.keys())
    if keys_surv != keys_fail:
        raise ValueError("Survival and failure rule dictionaries must have identical keys.")
    sys_st_list = sorted(keys_surv)
    max_st = max(sys_st_list)
    if sys_st_list != list(range(1, max_st + 1)):
        raise ValueError("Rule dictionary keys must be consecutive integers starting at 1.")

    # --- sampling loop (exactly n_sample draws) ---
    batch_size = max(1, min(int(n_batch), int(n_sample)))
    remaining = int(n_sample)
    counts = {s: 0 for s in [0] + sys_st_list}
    device = probs.device

    while remaining > 0:
        b = min(batch_size, remaining)
        samples = sample_categorical(probs_cond, b)  # (b, n_var, n_state) one-hot
        active = torch.ones(b, dtype=torch.bool, device=device)

        surv_prev = torch.ones(b, dtype=torch.bool, device=device) # survival indices in the previous rounds
        for s in range(1, max_st + 1):

            _res = classify_samples_with_indices(
                samples[active], rules_dict_surv[s], rules_dict_fail[s], return_masks=True
            )

            # back to original indices
            active_idx = torch.where(active)[0]  # positions in the original batch
            # subset masks from the classifier (length == active.sum())
            mask_surv_sub = _res["mask_survival"]
            mask_fail_sub = _res["mask_failure"]
            mask_unk_sub  = _res["mask_unknown"]

            # create full-size masks (length == b) and place subset masks at active positions
            mask_surv_full = torch.zeros(b, dtype=torch.bool, device=device)
            mask_fail_full = torch.zeros(b, dtype=torch.bool, device=device)
            mask_unk_full  = torch.zeros(b, dtype=torch.bool, device=device)

            mask_surv_full[active_idx] = mask_surv_sub
            mask_fail_full[active_idx] = mask_fail_sub
            mask_unk_full[active_idx]  = mask_unk_sub

            # Samples for sys = s-1
            _samp_s_1 = mask_fail_full & surv_prev
            counts[s-1] += int(_samp_s_1.sum().item())

            # update trackers
            active   = active & ~_samp_s_1  # remove finalized ones
            surv_prev = mask_surv_full # survivors roll to next level
        # Last state
        counts[s] += int(surv_prev.sum().item())
        active = active & ~surv_prev
        active_idx = torch.where(active)[0]  # positions in the original batch

        # Resolve unknowns with s_fun
        if active_idx.numel() > 0:

            for j in active_idx.tolist():
                sample_j = samples[j]  # (n_var, n_state)
                # convert one-hot row -> state index per var
                states = torch.argmax(sample_j, dim=1).tolist()

                # build comps dict for s_fun
                comps = {row_names[k]: int(states[k]) for k in range(n_comps)}

                _, sys_st, _ = s_fun(comps)
                counts[sys_st] += 1

        remaining -= b

    # --- normalize to probabilities (denominator = requested n_sample) ---
    total = float(n_sample)
    cond_probs = {k: counts[k] / total for k in counts}
    return cond_probs

def run_rule_extraction_by_mcs(
    *,
    sfun,
    probs: torch.Tensor,
    row_names: List[str],
    n_state: int,
    sys_surv_st: int,
    rules_surv: Optional[List[Dict[str, Any]]] = None,
    rules_fail: Optional[List[Dict[str, Any]]] = None,
    rules_mat_surv: Optional[torch.Tensor] = None,
    rules_mat_fail: Optional[torch.Tensor] = None,
    # Termination / threshold settings
    unk_prob_thres: float = 1e-2,
    unk_prob_opt: str = "rel", # "abs" or "rel"
    max_rounds: int = 10000,     # hard cap on rounds to prevent infinite loops
    # Frequencies / sampling settings
    prob_update_every: int = 500,
    save_every: int = 10,
    n_sample: int = 10_000_000,
    sample_batch_size: int = 100_000,
    max_search_loops: int = 0,  # max batches per round for searching unknowns (0 = use n_sample // sample_batch_size)
    min_rule_search: bool = True,
    rule_update_verbose: bool = True,
    # Biased sampling for rule discovery
    discovery_probs: Optional[torch.Tensor] = None,  # biased probs for search phase; true probs used for estimation
    bias_rounds: int = 0,  # use discovery_probs for first N rounds, then switch to true probs (0 = biased for all rounds)
    adaptive_bias: str = "",  # "alternating" or "gradient" or "" (off)
    adaptive_bias_phase_len: int = 100,  # rounds per phase for alternating; evaluation window for gradient
    adaptive_bias_hi: float = 10.0,  # high bias factor for alternating / max factor for gradient
    adaptive_bias_lo: float = 2.0,  # low bias factor for alternating / min factor for gradient
    # Boundary walking
    walk_every: int = 0,  # do boundary walks every N rounds (0 = off); e.g. 5 = walk on rounds 5,10,15,...
    walk_count: int = 1,  # number of walks per boundary-walk round
    # Parallelism
    n_workers: int = 1,  # number of CPU workers for parallel sfun + minimization
    devices: Optional[List[str]] = None,  # list of GPU devices for multi-GPU sampling, e.g. ["cuda:0", "cuda:1"]
    # Output control
    output_dir: str = "tsum_res",
    surv_json_name: str = None,
    fail_json_name: str = None,
    surv_pt_name: str = None,
    fail_pt_name: str = None,
    metrics_path: str = "metrics.json",
) -> Dict[str, Any]:

    os.makedirs(output_dir, exist_ok=True)

    if surv_json_name is None:
        surv_json_name = f"rules_geq_{sys_surv_st}.json"
    if fail_json_name is None:
        fail_json_name = f"rules_leq_{sys_surv_st-1}.json"
    if surv_pt_name is None:
        surv_pt_name = f"rules_geq_{sys_surv_st}.pt"
    if fail_pt_name is None:
        fail_pt_name = f"rules_leq_{sys_surv_st-1}.pt"

    # ---- helpers ----
    def _avg_rule_len(rule_store: Any) -> float:
        try:
            if not rule_store:
                return 0.0
            return (sum(len(r) - 1 for r in rule_store)) / len(rule_store)
        except Exception:
            return 0.0

    def _save_json(obj, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)

    def _save_pt(t: torch.Tensor, path: str) -> None:
        torch.save(t.detach().cpu(), path)

    # ---- initial state ----
    if rules_surv is None: rules_surv = []
    if rules_fail is None: rules_fail = []

    device = probs.device

    unk_prob = 1.0
    n_round = 0
    metrics_log: List[Dict[str, Any]] = []

    n_vars = len(row_names)
    if rules_mat_surv is None:
        if rules_surv:
            mats = [from_rule_dict_to_mat(r, row_names, n_state).to(device)
                    for r in rules_surv]
            rules_mat_surv = torch.stack(mats, dim=0)
            print(f"Loaded {len(rules_surv)} initial survival rules")
        else:
            rules_mat_surv = torch.empty((0, n_vars, n_state), dtype=torch.int32, device=device)
    if rules_mat_fail is None:
        if rules_fail:
            mats = [from_rule_dict_to_mat(r, row_names, n_state).to(device)
                    for r in rules_fail]
            rules_mat_fail = torch.stack(mats, dim=0)
            print(f"Loaded {len(rules_fail)} initial failure rules")
        else:
            rules_mat_fail = torch.empty((0, n_vars, n_state), dtype=torch.int32, device=device)

    sys_val_list: List[Any] = []

    metrics_path = os.path.join(output_dir, metrics_path)
    rules_surv_path = os.path.join(output_dir, surv_json_name)
    rules_fail_path = os.path.join(output_dir, fail_json_name)
    rules_surv_pt_path = os.path.join(output_dir, surv_pt_name)
    rules_fail_pt_path = os.path.join(output_dir, fail_pt_name)

    is_new_cand = True
    last_probs = {"survival": 0.0, "failure": 0.0, "unknown": 1.0}

    # ---- parallel worker pool (fork-based, inherits sfun via global) ----
    global _MP_SFUN, _MP_SYS_SURV_ST, _MP_N_STATE
    _pool = None
    if n_workers > 1:
        _MP_SFUN = sfun
        _MP_SYS_SURV_ST = sys_surv_st
        _MP_N_STATE = n_state
        _ctx = mp.get_context('fork')
        _pool = _ctx.Pool(n_workers)
        print(f"Parallel mode: {n_workers} CPU workers for sfun + minimization")

    # ---- biased search probs ----
    _using_biased = discovery_probs is not None
    search_probs = probs
    if _using_biased:
        assert discovery_probs.shape == probs.shape, (
            f"discovery_probs shape {discovery_probs.shape} != probs shape {probs.shape}")
        search_probs = discovery_probs.to(device)
        if bias_rounds > 0:
            print(f"Biased sampling: discovery_probs for first {bias_rounds} rounds, then true probs")
        else:
            print(f"Biased sampling: using discovery_probs for all rounds")

    # ---- adaptive bias state ----
    _adaptive_mode = adaptive_bias.lower().strip() if adaptive_bias else ""
    _adaptive_current_factor = 0.0
    _adaptive_phase_is_hi = True  # for alternating: start with high bias
    _adaptive_prev_unk = 1.0  # for gradient: track p_unk at last evaluation
    _adaptive_prev_round = 0  # for gradient: round at last evaluation
    _adaptive_last_delta_rate = 0.0  # for gradient: last Δp_unk/Δround rate
    if _adaptive_mode:
        if discovery_probs is None:
            # Auto-create initial discovery_probs from the hi factor
            _adaptive_current_factor = adaptive_bias_hi
            discovery_probs = make_discovery_probs(probs, bias_factor=adaptive_bias_hi)
            search_probs = discovery_probs.to(device)
            _using_biased = True
        else:
            _adaptive_current_factor = adaptive_bias_hi
        if _adaptive_mode == "alternating":
            print(f"Adaptive bias (alternating): hi={adaptive_bias_hi}, lo={adaptive_bias_lo}, "
                  f"phase={adaptive_bias_phase_len} rounds")
        elif _adaptive_mode == "gradient":
            print(f"Adaptive bias (gradient): range=[{adaptive_bias_lo}, {adaptive_bias_hi}], "
                  f"window={adaptive_bias_phase_len} rounds")
        else:
            print(f"WARNING: unknown adaptive_bias mode '{_adaptive_mode}', ignoring")
            _adaptive_mode = ""

    def _update_search_probs(new_factor):
        """Recompute search_probs with a new bias factor and update GPU copies."""
        nonlocal search_probs, _gpu_search_probs, _adaptive_current_factor
        _adaptive_current_factor = new_factor
        new_dp = make_discovery_probs(probs, bias_factor=new_factor)
        search_probs = new_dp.to(device)
        if _use_multi_gpu:
            _gpu_search_probs = [search_probs.to(d) for d in _gpu_devices]

    # ---- multi-GPU setup ----
    _use_multi_gpu = False
    _gpu_devices = []
    _gpu_probs = []       # probs replicated to each device (for estimation)
    _gpu_search_probs = []  # search probs replicated to each device
    _gpu_thread_pool = None
    if devices is not None and len(devices) > 1:
        from concurrent.futures import ThreadPoolExecutor
        _gpu_devices = [torch.device(d) for d in devices]
        _gpu_probs = [probs.to(d) for d in _gpu_devices]
        _gpu_search_probs = [search_probs.to(d) for d in _gpu_devices]
        _gpu_thread_pool = ThreadPoolExecutor(max_workers=len(_gpu_devices))
        _use_multi_gpu = True
        print(f"Multi-GPU mode: sampling across {devices}")

    total_loops = max(n_sample // sample_batch_size, 1)
    # Search loops: capped for finding unknowns; full total_loops used only for probability estimation
    search_loops = min(max_search_loops, total_loops) if max_search_loops > 0 else total_loops

    # ---- main loop ----
    while is_new_cand and (unk_prob > unk_prob_thres if unk_prob_opt == "abs" else unk_prob / (min([last_probs["failure"]+1e-12, last_probs["survival"]+1e-12])) > unk_prob_thres):
        n_round += 1
        t0 = time.perf_counter()

        # ---- auto-switch from biased to true probs ----
        if _using_biased and bias_rounds > 0 and n_round == bias_rounds + 1:
            search_probs = probs
            if _use_multi_gpu:
                _gpu_search_probs = [probs.to(d) for d in _gpu_devices]
            _using_biased = False
            print(f"*** Switching from biased to true probs at round {n_round} ***")

        # ---- adaptive bias adjustment ----
        if _adaptive_mode == "alternating" and _using_biased and n_round > 1:
            phase_round = (n_round - 1) % (2 * adaptive_bias_phase_len)
            should_be_hi = phase_round < adaptive_bias_phase_len
            if should_be_hi != _adaptive_phase_is_hi:
                new_factor = adaptive_bias_hi if should_be_hi else adaptive_bias_lo
                _update_search_probs(new_factor)
                _adaptive_phase_is_hi = should_be_hi
                print(f"*** Adaptive bias: switching to factor={new_factor:.1f} at round {n_round} ***")

        elif _adaptive_mode == "gradient" and _using_biased and n_round > 1:
            if (n_round - 1) % adaptive_bias_phase_len == 0 and n_round > adaptive_bias_phase_len:
                current_unk = last_probs.get("unknown", 1.0)
                delta_unk = _adaptive_prev_unk - current_unk  # positive = improving
                delta_rounds = n_round - _adaptive_prev_round
                rate = delta_unk / max(delta_rounds, 1)

                # Compare with previous rate to decide direction
                if _adaptive_last_delta_rate > 0 and rate < _adaptive_last_delta_rate:
                    # Got worse — reverse direction
                    if _adaptive_current_factor >= adaptive_bias_hi:
                        new_factor = max(adaptive_bias_lo, _adaptive_current_factor / 1.5)
                    elif _adaptive_current_factor <= adaptive_bias_lo:
                        new_factor = min(adaptive_bias_hi, _adaptive_current_factor * 1.5)
                    else:
                        # Was going up and it got worse, go down, or vice versa
                        new_factor = max(adaptive_bias_lo, _adaptive_current_factor / 1.5)
                else:
                    # Same or better — continue in same direction (slight push)
                    if rate > _adaptive_last_delta_rate:
                        # Keep current factor, it's working
                        new_factor = _adaptive_current_factor
                    else:
                        # Try a small change — alternate push direction
                        if _adaptive_current_factor > (adaptive_bias_hi + adaptive_bias_lo) / 2:
                            new_factor = max(adaptive_bias_lo, _adaptive_current_factor / 1.2)
                        else:
                            new_factor = min(adaptive_bias_hi, _adaptive_current_factor * 1.2)

                new_factor = max(adaptive_bias_lo, min(adaptive_bias_hi, new_factor))
                if abs(new_factor - _adaptive_current_factor) > 0.1:
                    _update_search_probs(new_factor)
                    print(f"*** Adaptive bias (gradient): factor {_adaptive_current_factor:.1f} -> {new_factor:.1f}, "
                          f"Δp_unk/round={rate:.2e} (prev={_adaptive_last_delta_rate:.2e}) ***")

                _adaptive_last_delta_rate = rate
                _adaptive_prev_unk = current_unk
                _adaptive_prev_round = n_round

        print("---")
        print(f"Round: {n_round}, Unk. prob.: {unk_prob:.3e}")
        if last_probs['survival'] is not None and last_probs['failure'] is not None:
            print(f"Surv probs: {last_probs['survival']:.3e}, Fail probs: {last_probs['failure']:.3e}")
        print(f"No. of non-dominant rules: {len(rules_mat_surv)+len(rules_mat_fail)}, "
              f"Survival rules: {len(rules_mat_surv)}, Failure rules: {len(rules_mat_fail)}")

        is_new_cand = False
        counts = {"survival": 0, "failure": 0, "unknown": 0}
        res = None
        samples = None
        i = -1
        _t_search = 0.0
        _t_minimize = 0.0
        _t_rules = 0.0
        _t_probs = 0.0
        _walk_used = False

        # ---- boundary walk round ----
        if walk_every > 0 and n_round % walk_every == 0:
            _ts = time.perf_counter()
            walk_results = boundary_walk(
                sfun=sfun,
                row_names=row_names,
                n_state=n_state,
                sys_surv_st=sys_surv_st,
                probs=probs,
                rules_mat_surv=rules_mat_surv,
                rules_mat_fail=rules_mat_fail,
                n_walks=max(walk_count, n_workers) if _pool else walk_count,
            )
            _t_search = time.perf_counter() - _ts

            if walk_results:
                _walk_used = True
                is_new_cand = True
                print(f"Boundary walk: {len(walk_results)} unknown sample(s) found")

                _ts = time.perf_counter()
                new_surv_dicts = []
                new_fail_dicts = []

                if _pool:
                    # Parallel minimization of walk results
                    tasks = []
                    for comps_st, fval_w, sys_st_w in walk_results:
                        tasks.append((comps_st, fval_w))
                    min_results = _pool.map(_minimize_one_unknown, tasks)

                    for min_comps_st, sys_st_m, fval_m in min_results:
                        if sys_st_m >= sys_surv_st:
                            new_surv_dicts.append(min_comps_st)
                        else:
                            new_fail_dicts.append(min_comps_st)
                        if isinstance(fval_m, float):
                            fval_m = int(round(fval_m * 1000)) / 1000.0
                        if fval_m not in sys_val_list:
                            sys_val_list.append(fval_m)
                            sys_val_list.sort(key=mixed_sort_key)
                else:
                    # Serial minimization
                    for comps_st, fval_w, sys_st_w in walk_results:
                        if sys_st_w >= sys_surv_st:
                            min_comps_st, info = minimise_surv_states_random(
                                comps_st, sfun, sys_surv_st=sys_surv_st, fval=fval_w)
                            fval_w = info.get('final_sys_state', fval_w)
                            new_surv_dicts.append(min_comps_st)
                        else:
                            min_comps_st, info = minimise_fail_states_random(
                                comps_st, sfun, max_state=n_state - 1,
                                sys_fail_st=sys_surv_st - 1, fval=fval_w)
                            fval_w = info.get('final_sys_state', fval_w)
                            new_fail_dicts.append(min_comps_st)
                        if isinstance(fval_w, float):
                            fval_w = int(round(fval_w * 1000)) / 1000.0
                        if fval_w not in sys_val_list:
                            sys_val_list.append(fval_w)
                            sys_val_list.sort(key=mixed_sort_key)

                _t_minimize = time.perf_counter() - _ts
                _ts = time.perf_counter()

                if new_surv_dicts:
                    rules_surv, rules_mat_surv, n_add, n_rem = update_rules_batch(
                        new_surv_dicts, rules_surv, rules_mat_surv, row_names, verbose=rule_update_verbose)
                    print(f"Walk survival: {n_add} rules added, {n_rem} removed (from {len(new_surv_dicts)} candidates)")
                if new_fail_dicts:
                    rules_fail, rules_mat_fail, n_add, n_rem = update_rules_batch(
                        new_fail_dicts, rules_fail, rules_mat_fail, row_names, verbose=rule_update_verbose)
                    print(f"Walk failure: {n_add} rules added, {n_rem} removed (from {len(new_fail_dicts)} candidates)")

                if sys_val_list:
                    sys_val_list.sort(key=mixed_sort_key)
                _t_rules = time.perf_counter() - _ts

                # ---- metrics and save for walk round ----
                n_sample_actual = 0
                probs_updated = False
                _t_probs = 0.0
                if (n_round % prob_update_every) == 0:
                    _ts_p = time.perf_counter()
                    loops = max(n_sample // sample_batch_size, 1)
                    c2 = {"survival": 0, "failure": 0, "unknown": 0}
                    for _ in range(loops):
                        if _use_multi_gpu:
                            n_gpus = len(_gpu_devices)
                            per_gpu = sample_batch_size // n_gpus
                            remainder = sample_batch_size % n_gpus
                            tasks = []
                            for gi in range(n_gpus):
                                n_gi = per_gpu + (1 if gi < remainder else 0)
                                rules_s_gi = rules_mat_surv.to(_gpu_devices[gi])
                                rules_f_gi = rules_mat_fail.to(_gpu_devices[gi])
                                tasks.append((_gpu_probs[gi], n_gi, rules_s_gi, rules_f_gi, False))
                            for _, ci in _gpu_thread_pool.map(_sample_and_classify_on_device, tasks):
                                for k in c2:
                                    c2[k] += ci[k]
                        else:
                            s = sample_categorical(probs, sample_batch_size)
                            ci = classify_samples(s, rules_mat_surv, rules_mat_fail)
                            for k in c2:
                                c2[k] += ci[k]
                    sp2 = {k: v / (sample_batch_size * loops) for k, v in c2.items()}
                    print(f"Probs: 'surv': {sp2['survival']: .3e}, 'fail': {sp2['failure']: .3e}, 'unkn': {sp2['unknown']: .3e}")
                    unk_prob = sp2["unknown"]
                    last_probs.update(sp2)
                    n_sample_actual = sample_batch_size * loops
                    probs_updated = True
                    _t_probs = time.perf_counter() - _ts_p

                dt = time.perf_counter() - t0
                rss_gb = psutil.Process().memory_info().rss / (1024**3)
                metrics_log.append({
                    "round": n_round,
                    "time_sec": dt,
                    "t_search": round(_t_search, 3),
                    "t_minimize": round(_t_minimize, 3),
                    "t_rules": round(_t_rules, 3),
                    "t_probs": round(_t_probs, 3),
                    "n_rules_surv": int(len(rules_mat_surv)),
                    "n_rules_fail": int(len(rules_mat_fail)),
                    "probs_updated": probs_updated,
                    "p_survival": last_probs["survival"],
                    "p_failure": last_probs["failure"],
                    "p_unknown": last_probs["unknown"],
                    "n_sample_actual": n_sample_actual,
                    "avg_len_surv": _avg_rule_len(rules_surv),
                    "avg_len_fail": _avg_rule_len(rules_fail),
                    "rss_gb": rss_gb,
                    "walk_round": True,
                    **({"bias_factor": _adaptive_current_factor} if _adaptive_mode else {}),
                })

                if (n_round % save_every) == 0:
                    with open(metrics_path, "a", encoding="utf-8") as mf:
                        for e in metrics_log[-save_every:]:
                            mf.write(json.dumps(e) + "\n")
                    _save_json(rules_surv, rules_surv_path)
                    _save_json(rules_fail, rules_fail_path)
                    _save_pt(rules_mat_surv, rules_surv_pt_path)
                    _save_pt(rules_mat_fail, rules_fail_pt_path)

                if n_round >= max_rounds:
                    print(f"Reached maximum rounds ({max_rounds}). Terminating.")
                    break
                continue  # skip normal search for this round

        # ---- normal sampling search ----
        _ts = time.perf_counter()
        for i in range(search_loops):
            if _use_multi_gpu:
                # Split batch across GPUs, sample + classify in parallel threads
                n_gpus = len(_gpu_devices)
                per_gpu = sample_batch_size // n_gpus
                remainder = sample_batch_size % n_gpus
                tasks = []
                for gi in range(n_gpus):
                    n_gi = per_gpu + (1 if gi < remainder else 0)
                    rules_s_gi = rules_mat_surv.to(_gpu_devices[gi])
                    rules_f_gi = rules_mat_fail.to(_gpu_devices[gi])
                    tasks.append((_gpu_search_probs[gi], n_gi, rules_s_gi, rules_f_gi, True))

                futures = list(_gpu_thread_pool.map(_sample_and_classify_on_device, tasks))

                # Merge results back to primary device
                all_samples = []
                for samples_gi, res_gi in futures:
                    all_samples.append(samples_gi.to(device))
                    counts["survival"] += int(res_gi["survival"])
                    counts["failure"]  += int(res_gi["failure"])
                    counts["unknown"]  += int(res_gi["unknown"])
                samples = torch.cat(all_samples, dim=0)

                # Re-classify merged batch on primary device for correct indices
                res = classify_samples_with_indices(samples, rules_mat_surv, rules_mat_fail, return_masks=True)
            else:
                samples = sample_categorical(search_probs, sample_batch_size)  # (B, n_var, n_state)
                res = classify_samples_with_indices(samples, rules_mat_surv, rules_mat_fail, return_masks=True)

                counts["survival"] += int(res["survival"])
                counts["failure"]  += int(res["failure"])
                counts["unknown"]  += int(res["unknown"])

            if res['idx_unknown'].numel() > 0:
                is_new_cand = True
                break

        _t_search = time.perf_counter() - _ts

        # denominator = number of samples actually processed
        n_sample_actual = sample_batch_size * (i + 1)
        # When using biased search probs, the counts don't reflect true probabilities.
        # Use last_probs from previous estimation; only update from true-probs estimation.
        if not _using_biased:
            samp_probs = {k: v / n_sample_actual for k, v in counts.items()}
            unk_prob = samp_probs["unknown"]
            last_probs.update(samp_probs)

        # If no unknowns found, skip candidate creation and continue to periodic update / exit
        if not is_new_cand:
            probs_updated = False
            # When search is capped, the unk_prob estimate from search_loops is rough;
            # force a full probability update to get an accurate termination check.
            # Also force when using biased discovery_probs (search counts don't reflect true probs).
            needs_full_estimate = _using_biased or (search_loops < total_loops) or (n_round % prob_update_every) == 0
            if needs_full_estimate:
                # refresh with a full estimate
                loops = max(n_sample // sample_batch_size, 1)
                c2 = {"survival": 0, "failure": 0, "unknown": 0}
                for _ in range(loops):
                    if _use_multi_gpu:
                        n_gpus = len(_gpu_devices)
                        per_gpu = sample_batch_size // n_gpus
                        remainder = sample_batch_size % n_gpus
                        tasks = []
                        for gi in range(n_gpus):
                            n_gi = per_gpu + (1 if gi < remainder else 0)
                            rules_s_gi = rules_mat_surv.to(_gpu_devices[gi])
                            rules_f_gi = rules_mat_fail.to(_gpu_devices[gi])
                            tasks.append((_gpu_probs[gi], n_gi, rules_s_gi, rules_f_gi, False))
                        for _, ci in _gpu_thread_pool.map(_sample_and_classify_on_device, tasks):
                            for k in c2:
                                c2[k] += ci[k]
                    else:
                        s = sample_categorical(probs, sample_batch_size)
                        ci = classify_samples(s, rules_mat_surv, rules_mat_fail)
                        for k in c2:
                            c2[k] += ci[k]
                sp2 = {k: v / (sample_batch_size * loops) for k, v in c2.items()}
                print("---")
                print(f"Probs: 'surv': {sp2['survival']: .3e}, 'fail': {sp2['failure']: .3e}, 'unkn': {sp2['unknown']: .3e}")
                unk_prob = sp2["unknown"]
                last_probs.update(sp2)
                n_sample_actual = sample_batch_size * loops
                probs_updated = True

            # metrics, persist, then break condition handled by while guard
            dt = time.perf_counter() - t0
            rss_gb = psutil.Process().memory_info().rss / (1024**3)
            metrics_log.append({
                "round": n_round,
                "time_sec": dt,
                "t_search": round(_t_search, 3),
                "t_minimize": 0.0,
                "t_rules": 0.0,
                "t_probs": round(dt - _t_search, 3),
                "n_rules_surv": int(len(rules_mat_surv)),
                "n_rules_fail": int(len(rules_mat_fail)),
                "probs_updated": probs_updated,
                "p_survival": last_probs["survival"],
                "p_failure": last_probs["failure"],
                "p_unknown": last_probs["unknown"],
                "n_sample_actual": n_sample_actual,
                "avg_len_surv": _avg_rule_len(rules_surv),
                "avg_len_fail": _avg_rule_len(rules_fail),
                "rss_gb": rss_gb,
                **({"bias_factor": _adaptive_current_factor} if _adaptive_mode else {}),
            })

            if (n_round % save_every) == 0:
                with open(metrics_path, "a", encoding="utf-8") as mf:
                    for e in metrics_log[-save_every:]:
                        mf.write(json.dumps(e) + "\n")
                _save_json(rules_surv, rules_surv_path)
                _save_json(rules_fail, rules_fail_path)
                _save_pt(rules_mat_surv, rules_surv_pt_path)
                _save_pt(rules_mat_fail, rules_fail_pt_path)

            continue  # go to next while-check (likely exit if unk_prob <= thresh)

        # --- We have unknowns: extract unknown(s) and build rule(s) ---
        idx_unknown = res['idx_unknown']

        _ts = time.perf_counter()
        if _pool is not None and min_rule_search:
            # ---- Parallel: pick up to n_workers unknowns and minimize concurrently ----
            n_pick = min(n_workers, len(idx_unknown))
            perm = torch.randperm(len(idx_unknown))[:n_pick]
            picked_indices = idx_unknown[perm]

            tasks = []
            for idx_i in picked_indices:
                s0 = samples[idx_i.item()]
                sts = torch.argmax(s0, dim=1).tolist()
                cst = {row_names[k]: int(sts[k]) for k in range(n_vars)}
                tasks.append((cst, None))

            results = _pool.map(_minimize_one_unknown, tasks)
            _t_minimize = time.perf_counter() - _ts

            _ts = time.perf_counter()
            # Separate results into survival and failure batches
            new_surv_dicts = []
            new_fail_dicts = []
            for min_comps_st, sys_st, fval in results:
                if sys_st >= sys_surv_st:
                    new_surv_dicts.append(min_comps_st)
                else:
                    new_fail_dicts.append(min_comps_st)

                if isinstance(fval, float):
                    fval = int(round(fval * 1000)) / 1000.0
                if fval not in sys_val_list:
                    sys_val_list.append(fval)
                    sys_val_list.sort(key=mixed_sort_key)

            # Batch update: one dominance check per type instead of N sequential ones
            if new_surv_dicts:
                rules_surv, rules_mat_surv, n_add, n_rem = update_rules_batch(
                    new_surv_dicts, rules_surv, rules_mat_surv, row_names, verbose=rule_update_verbose)
                print(f"Survival: {n_add} rules added, {n_rem} removed (from {len(new_surv_dicts)} candidates)")
            if new_fail_dicts:
                rules_fail, rules_mat_fail, n_add, n_rem = update_rules_batch(
                    new_fail_dicts, rules_fail, rules_mat_fail, row_names, verbose=rule_update_verbose)
                print(f"Failure: {n_add} rules added, {n_rem} removed (from {len(new_fail_dicts)} candidates)")

            if sys_val_list:
                sys_val_list.sort(key=mixed_sort_key)
                print(f"Updated sys_vals: {sys_val_list}")

        else:
            # ---- Serial (original): pick one unknown ----
            rand_idx = idx_unknown[torch.randint(len(idx_unknown), (1,))].item()
            sample0 = samples[rand_idx]  # (n_var, n_state)

            states = torch.argmax(sample0, dim=1).tolist()
            comps_st_test = {row_names[k]: int(states[k]) for k in range(n_vars)}

            fval, sys_st, min_comps_st0 = sfun(comps_st_test)
            if min_comps_st0 is None:
                min_comps_st0 = comps_st_test.copy()
            elif isinstance(next(iter(min_comps_st0.values())), tuple):
                min_comps_st0 = {k: v[1] for k, v in min_comps_st0.items()}

            if sys_st >= sys_surv_st:
                if min_rule_search:
                    min_comps_st, info = minimise_surv_states_random(min_comps_st0, sfun, sys_surv_st=sys_surv_st, fval=fval)
                    fval = info.get('final_sys_state', fval)
                else:
                    min_comps_st = get_min_surv_comps_st(min_comps_st0, sys_surv_st=sys_surv_st)
            else:
                if min_rule_search:
                    min_comps_st, info = minimise_fail_states_random(min_comps_st0, sfun, max_state=n_state-1, sys_fail_st=sys_surv_st-1, fval=fval)
                    fval = info.get('final_sys_state', fval)
                else:
                    min_comps_st = get_min_fail_comps_st(min_comps_st0, max_st=n_state-1, sys_fail_st=sys_surv_st-1)
            _t_minimize = time.perf_counter() - _ts

            _ts = time.perf_counter()
            if sys_st >= sys_surv_st:
                print("Survival sample found from sampling.")
                rules_surv, rules_mat_surv = update_rules(min_comps_st, rules_surv, rules_mat_surv, row_names, verbose=rule_update_verbose)
            else:
                print("Failure sample found from sampling.")
                rules_fail, rules_mat_fail = update_rules(min_comps_st, rules_fail, rules_mat_fail, row_names, verbose=rule_update_verbose)

            print(f"New rule added. System state: {sys_st}, System value: {fval}. Total samples: {n_sample_actual}.")
            print(f"New rule (No. of conditions: {len(min_comps_st)-1}): {min_comps_st}")

            if isinstance(fval, float):
                fval = int(round(fval * 1000)) / 1000.0
            if fval not in sys_val_list:
                sys_val_list.append(fval)
                sys_val_list.sort(key=mixed_sort_key)
                print(f"Updated sys_vals: {sys_val_list}")

        # ---- Periodic probability (bound) test via sampling ----
        if _t_rules == 0.0:
            _t_rules = time.perf_counter() - _ts
        probs_updated = False
        _ts = time.perf_counter()
        if (n_round % prob_update_every) == 0:
            loops = max(n_sample // sample_batch_size, 1)
            c2 = {"survival": 0, "failure": 0, "unknown": 0}
            for _ in range(loops):
                if _use_multi_gpu:
                    n_gpus = len(_gpu_devices)
                    per_gpu = sample_batch_size // n_gpus
                    remainder = sample_batch_size % n_gpus
                    tasks = []
                    for gi in range(n_gpus):
                        n_gi = per_gpu + (1 if gi < remainder else 0)
                        rules_s_gi = rules_mat_surv.to(_gpu_devices[gi])
                        rules_f_gi = rules_mat_fail.to(_gpu_devices[gi])
                        tasks.append((_gpu_probs[gi], n_gi, rules_s_gi, rules_f_gi, False))
                    for _, ci in _gpu_thread_pool.map(_sample_and_classify_on_device, tasks):
                        for k in c2:
                            c2[k] += ci[k]
                else:
                    s = sample_categorical(probs, sample_batch_size)
                    ci = classify_samples(s, rules_mat_surv, rules_mat_fail)
                    for k in c2:
                        c2[k] += ci[k]
            sp2 = {k: v / (sample_batch_size * loops) for k, v in c2.items()}
            print("---")
            print(f"Probs: 'surv': {sp2['survival']: .3e}, 'fail': {sp2['failure']: .3e}, 'unkn': {sp2['unknown']: .3e}")
            unk_prob = sp2["unknown"]
            last_probs.update(sp2)
            n_sample_actual = sample_batch_size * loops
            probs_updated = True

        # ---- metrics for this round ----
        _t_probs = time.perf_counter() - _ts
        rss_gb = psutil.Process().memory_info().rss / (1024**3)
        dt = time.perf_counter() - t0
        metrics_log.append({
            "round": n_round,
            "time_sec": dt,
            "t_search": round(_t_search, 3),
            "t_minimize": round(_t_minimize, 3),
            "t_rules": round(_t_rules, 3),
            "t_probs": round(_t_probs, 3),
            "n_rules_surv": int(len(rules_mat_surv)),
            "n_rules_fail": int(len(rules_mat_fail)),
            "probs_updated": probs_updated,
            "p_survival": last_probs["survival"],
            "p_failure": last_probs["failure"],
            "p_unknown": last_probs["unknown"],
            "n_sample_actual": n_sample_actual,
            "avg_len_surv": _avg_rule_len(rules_surv),
            "avg_len_fail": _avg_rule_len(rules_fail),
            "rss_gb": rss_gb,
        })

        if (n_round % save_every) == 0:
            with open(metrics_path, "a", encoding="utf-8") as mf:
                for e in metrics_log[-save_every:]:
                    mf.write(json.dumps(e) + "\n")
            _save_json(rules_surv, rules_surv_path)
            _save_json(rules_fail, rules_fail_path)
            _save_pt(rules_mat_surv, rules_surv_pt_path)
            _save_pt(rules_mat_fail, rules_fail_pt_path)

        if n_round >= max_rounds:
            print(f"Reached maximum rounds ({max_rounds}). Terminating.")
            break

    # Final flush of any remaining metrics not yet written by save_every
    last_flushed_rounds = (n_round // save_every) * save_every
    if last_flushed_rounds < n_round and metrics_log:
        with open(metrics_path, "a", encoding="utf-8") as mf:
            for e in metrics_log[last_flushed_rounds:]:
                mf.write(json.dumps(e) + "\n")

    # Final snapshot of rules
    _save_json(rules_surv, rules_surv_path)
    _save_json(rules_fail, rules_fail_path)
    _save_pt(rules_mat_surv, rules_surv_pt_path)
    _save_pt(rules_mat_fail, rules_fail_pt_path)

    # Final probability check
    loops = max(n_sample // sample_batch_size, 1)
    c2 = {"survival": 0, "failure": 0, "unknown": 0}
    for _ in range(loops):
        if _use_multi_gpu:
            n_gpus = len(_gpu_devices)
            per_gpu = sample_batch_size // n_gpus
            remainder = sample_batch_size % n_gpus
            tasks = []
            for gi in range(n_gpus):
                n_gi = per_gpu + (1 if gi < remainder else 0)
                rules_s_gi = rules_mat_surv.to(_gpu_devices[gi])
                rules_f_gi = rules_mat_fail.to(_gpu_devices[gi])
                tasks.append((_gpu_probs[gi], n_gi, rules_s_gi, rules_f_gi, False))
            for _, ci in _gpu_thread_pool.map(_sample_and_classify_on_device, tasks):
                for k in c2:
                    c2[k] += ci[k]
        else:
            s = sample_categorical(probs, sample_batch_size)
            ci = classify_samples(s, rules_mat_surv, rules_mat_fail)
            for k in c2:
                c2[k] += ci[k]
    sp2 = {k: v / (sample_batch_size * loops) for k, v in c2.items()}
    print("---")
    print(f"[Final results] Probs: 'surv': {sp2['survival']: .3e}, 'fail': {sp2['failure']: .3e}, 'unkn': {sp2['unknown']: .3e}")

    # Final metrics entry
    rss_gb = psutil.Process().memory_info().rss / (1024**3)
    metrics_log.append({
        "round": n_round,
        "time_sec": 0.0,
        "n_rules_surv": int(len(rules_mat_surv)),
        "n_rules_fail": int(len(rules_mat_fail)),
        "probs_updated": True,
        "p_survival": sp2["survival"],
        "p_failure": sp2["failure"],
        "p_unknown": sp2["unknown"],
        "avg_len_surv": _avg_rule_len(rules_surv),
        "avg_len_fail": _avg_rule_len(rules_fail),
        "rss_gb": rss_gb,   
    })

    # ---- clean up worker pools ----
    if _pool is not None:
        _pool.close()
        _pool.join()
    if _gpu_thread_pool is not None:
        _gpu_thread_pool.shutdown(wait=False)

    return {
        "sys_vals": sorted(sys_val_list, key=mixed_sort_key),
        "metrics_path": metrics_path,
        "rules_surv_path": rules_surv_path,
        "rules_fail_path": rules_fail_path,
        "rules_surv_pt_path": rules_surv_pt_path,
        "rules_fail_pt_path": rules_fail_pt_path,
        "metrics_log": metrics_log,
    }

