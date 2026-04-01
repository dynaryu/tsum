"""
Shared DC-OPF blackout model for TSUM demos.

Pure Python DC-OPF solver (scipy/HiGHS, no MATLAB/Octave dependency)
compatible with MATPOWER .m case files.

Usage:
    from dcopt import make_dcopt_sfun

    sfun = make_dcopt_sfun(
        case_path='case14.m',
        blackout_threshold=54.8,
        alpha=2.0,
    )
    fval, sys_st, _ = sfun(comps_st)

For direct solver access:
    from dcopt.func_dcopt_py import load_case, add_branch_capacity, func_dcopt, DcopfPrecomputed
"""

from .sfun_dcopt import make_dcopt_sfun
from .func_dcopt_py import (
    load_case,
    add_branch_capacity,
    load2disp,
    func_dcopt,
    DcopfPrecomputed,
)
