"""
TORCH: Transient Omni-dimensional Reactive Computational Hydrodynamics.
Finite Volume 1D/2D/3D Euler Solver with MUSCL, HLLC, Arrhenius/ZND kinetics,
Exact Riemann validation, AMR indicators, and optional standalone JIT kernels.

Unified python package:
- torchSolver: core thermodynamics, limiters, Riemann solver, MUSCL, HLLC, kinetics, presets, simulation loops.
- torchAnalysis: exact Riemann solution, AMR indicators, convergence study, Navier-Stokes fluxes, CSV plotting.
"""

from .torchSolver import (
    run_simulation,
    run_simulation_wqr,
    run_simulation_wQR,
    compute_conservation_audit,
    PRESETS,
    get_preset,
    PresetConfig,
    SimulationPreset,
    uconsv_to_uprim,
    uprim_to_uconsv,
    temp_calc,
    gamma_calc,
    sound_speed_calc,
    bc_creator,
    slope_limiter_calc,
    wave_speed_estimate,
    pressure_estimate,
    muscl_application,
    muscl_application_wqr,
    muscl_wlim_calc,
    deltai_calc,
    hllc_flux_calc_for_muscl,
    delta_t_estimate,
    flux_application_05t,
    new_lambda_n_e_calc,
    jit_vanleer_limiter,
    jit_state_update,
    NUMBA_AVAILABLE,
)

from .torchAnalysis import (
    exact_riemann_sod,
    compute_error_norms,
    compute_gradient_indicator,
    get_amr_refinement_mask,
    run_convergence_study,
    compute_viscous_heat_fluxes,
    plot_csv,
)

from .torchSolver2D import (
    run_torch_solver_2d,
    get_2d_preset,
    uconsv_to_uprim_2d,
    uprim_to_uconsv_2d,
    sound_speed_calc_2d,
    temp_calc_2d,
    gamma_calc_2d,
    muscl_hancock_step_2d,
    hllc_flux_1d_slice,
    compute_tvd_slopes_2d,
    apply_boundary_conditions_2d,
    reaction_step_2d,
    compute_cfl_dt_2d,
)

from .torchSolver3D import (
    run_torch_solver_3d, get_3d_preset, PRESETS_3D,
    uprim_to_uconsv_3d, uconsv_to_uprim_3d,
    apply_boundary_conditions_3d, muscl_hancock_step_3d, reaction_step_3d,
)

# Backward compatibility module aliases
from . import torchSolver as numerics
from . import torchSolver as solver
from . import torchSolver as kinetics
from . import torchSolver as presets
from . import torchSolver as jitAccel
from . import torchAnalysis as analysis

__all__ = [
    # Solvers & loops
    "run_simulation",
    "run_simulation_wqr",
    "run_simulation_wQR",
    "compute_conservation_audit",
    # Presets
    "PRESETS",
    "get_preset",
    "PresetConfig",
    "SimulationPreset",
    # Physics & Numerics
    "uconsv_to_uprim",
    "uprim_to_uconsv",
    "temp_calc",
    "gamma_calc",
    "sound_speed_calc",
    "bc_creator",
    "slope_limiter_calc",
    "wave_speed_estimate",
    "pressure_estimate",
    "muscl_application",
    "muscl_application_wqr",
    "muscl_wlim_calc",
    "deltai_calc",
    "hllc_flux_calc_for_muscl",
    "delta_t_estimate",
    "flux_application_05t",
    "new_lambda_n_e_calc",
    # JIT kernels
    "jit_vanleer_limiter",
    "jit_state_update",
    "NUMBA_AVAILABLE",
    # Analysis & Plotting
    "exact_riemann_sod",
    "compute_error_norms",
    "compute_gradient_indicator",
    "get_amr_refinement_mask",
    "run_convergence_study",
    "compute_viscous_heat_fluxes",
    "plot_csv",
    # 2D Solver & Physics
    "run_torch_solver_2d",
    "get_2d_preset",
    "uconsv_to_uprim_2d",
    "uprim_to_uconsv_2d",
    "sound_speed_calc_2d",
    "temp_calc_2d",
    "gamma_calc_2d",
    "muscl_hancock_step_2d",
    "hllc_flux_1d_slice",
    "compute_tvd_slopes_2d",
    "apply_boundary_conditions_2d",
    "reaction_step_2d",
    "compute_cfl_dt_2d",
    # 3D Solver & Physics
    "run_torch_solver_3d", "get_3d_preset", "PRESETS_3D",
    "uprim_to_uconsv_3d", "uconsv_to_uprim_3d",
    "apply_boundary_conditions_3d", "muscl_hancock_step_3d", "reaction_step_3d",
    # Backward compat
    "numerics",
    "solver",
    "kinetics",
    "presets",
    "jitAccel",
    "analysis",
]

from .torchSession import Simulation

__all__ += ["Simulation"]
