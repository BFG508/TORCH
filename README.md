# TORCH 🔥

**T**ransient **O**mni-dimensional **R**eactive **C**omputational **H**ydrodynamics

An experimental Python and MATLAB framework for compressible flow and reduced reactive gas dynamics in **1D, 2D and 3D**. This project numerically solves the Euler equations using finite-volume methods, combining shock-tube benchmarks, reactive-flow simulations and interactive visualization with conservation diagnostics and numerical verification.

Current release: **v1.4.0** — independent analytic verification, reproducible benchmarks, reduced solver work storage and corrected 1D MUSCL/HLLC consistency.

## 🚀 Features

* **Finite-Volume Hydrodynamics:** Solves the conservative Euler equations with MUSCL-Hancock reconstruction and HLLC interface fluxes, including tangential momentum transport in 2D and both tangential momenta in 3D.
* **Reactive Gas Dynamics:** Includes Arrhenius chemistry, composition-dependent thermodynamics and conservative reactant transport. Python also supports a reduced induction/exothermic two-stage model through `--kinetic_model znd`.
* **Slope Limiters & Boundary Conditions:** Provides Van Leer, Van Albada, MINBEE, Superbee and bounded Unos limiters, alongside open, reflective and periodic boundaries.
* **Benchmark Simulations:** Includes Sod shock tubes, rarefaction waves, strong shocks, 2D explosions and implosions, quadrant Riemann problems, a reduced annular RDE chamber case, 3D Sod tubes along each axis, spherical explosions and 3D ignition.
* **Interactive Visualization:** Includes the original 1D interfaces and session desktops for 2D/3D in both languages. Explore maps, physical 3D slices and profiles, pause/resume simulations, save/load restarts and replay recorded frames from disk. The original Python 1D interface retains its exact Riemann overlay.
* **Data Export & Diagnostics:** Exports CSV datasets, JSON metadata and PNG/SVG figures through the Python workflows, with boundary-aware conservation audits, five-equation volume balances in 3D and a 2D exit-thrust diagnostic per unit depth.
* **Orthogonal Geometries & Face Boundaries:** Adds physical cell volumes/face areas, local-basis momentum sources and per-face prescribed, reservoir-inlet and pressure-outlet conditions through the session APIs and CLI.
* **Checkpoints & Restart:** Adds bounded-memory simulation sessions in 1D, 2D and 3D, versioned HDF5 checkpoints and Python/MATLAB restart interchange for Arrhenius and inert cases. Python ZND checkpoints preserve the induction variable.
* **Temporal Scientific Output:** Streams states to disk with Cartesian/curved VTK grids, PVD time collections and XDMF/HDF5 snapshots or time collections, suitable for external scientific viewers.
* **Numerical Verification:** Includes exact Riemann comparisons, analytic vortex/acoustic/shock mesh studies, Python and MATLAB/Octave tests, and cross-language regression checks with JSON reports.
* **Performance & Memory:** Batches 2D/3D HLLC work, avoids unused reconstruction temporaries, reuses one MATLAB metric grid and stores its indices as compact integers. Benchmark tools report warmed timings and explicitly scoped memory measurements.

The current version supports **1D, 2D and 3D** on Cartesian grids and supported cylindrical, annular and spherical domains through sessions. Desktop workflows cover all three dimensions: original interfaces handle 1D, and the v1.3 session desktops create 2D/3D simulations. The CLI and function APIs remain available. The chemistry and RDE boundary model are reduced approximations; quantitative detonation predictions require further physical calibration and validation. Hydrodynamic reconstruction is second order for smooth solutions, while species transport is first order. Equations, units and accuracy limits are documented below.

## 🛠️ Technology Stack

This project is developed in **Python** and **MATLAB**. The Python implementation uses **NumPy** for numerical operations, **Matplotlib** for visualization and **Tkinter** for its desktop interface. **Numba** supplies optional standalone acceleration kernels; the simulation drivers do not automatically use them.

Python **3.10 or newer** is required. The numerical MATLAB functions are also tested with **GNU Octave**, while the `.mlapp` desktop interface requires MATLAB App Designer. HDF5 checkpoint and XDMF output use **h5py** in Python and MATLAB’s native HDF5 functions. Python dependencies are listed in `requirements.txt`; Tkinter may require installation through the operating system.

## 📂 Repository Structure

* `python/` - Python simulation drivers, visualization tools and numerical package.
  * `cfdSolver/torchSolver.py` - Contains the 1D hydrodynamics, thermodynamics, reaction models and simulation presets.
  * `cfdSolver/torchSolver2D.py` - Implements the 2D solver, reactive driver, boundary conditions and RDE diagnostics.
  * `cfdSolver/torchGeometry.py` / `torchBoundaries.py` / `torchFiniteVolume.py` - Define orthogonal metrics, per-face conditions and the MUSCL–Rusanov/SSPRK2 session kernel.
  * `cfdSolver/torchSession.py` - Executes resumable sessions one complete step at a time in all dimensions.
  * `cfdSolver/torchIO.py` - Handles validated HDF5 checkpoints, VTK/XDMF exports and streaming time collections.
  * `cfdSolver/torchSolver3D.py` - Implements the Cartesian 3D solver, reaction/transport coupling and volume balances.
  * `cfdSolver/torchFlux.py` - Shares directional HLLC fluxes between the 2D and 3D solvers.
  * `cfdSolver/torchPlot3D.py` - Exports 3D volume slices and centerline profiles.
  * `cfdSolver/torchAnalysis.py` - Provides exact Riemann references, error norms, convergence studies and figure exports.
  * `cfdSolver/torchVerification.py` - Defines independent analytic references and mesh-study reports.
  * **Simulation & Visualization Tools:**
    * `torchCli.py` - Runs batch simulations and exports CSV, JSON and figures; adds checkpoint, restart and temporal-output options.
    * `torchSessionCli.py` - Adapts session configuration and scientific outputs to the CLI.
    * `torchGui.py` - Launches the original 1D interface by default; `--dim 2d` / `--dim 3d` open the session desktop.
    * `torchDesktop.py` / `desktop/` - Provide the 2D/3D desktop, validated setup, single-owner worker, disk playback and coordinate-aware plots.
    * `torchPlot.py` - Exports figures from existing simulation CSV files.
    * `torchVerify.py` / `torchBenchmark.py` - Run analytic mesh studies and repeatable timing/allocation measurements.
  * `tests/` - Contains numerical, command-line and graphical regression tests.
* `matlab/` - MATLAB implementation and compatibility drivers.
  * `setupTorch.m` - Adds the runtime subfolders to the MATLAB/Octave path from any working directory.
  * `solvers/` - Contains the 1D, 2D and 3D finite-volume solvers and directional HLLC fluxes.
  * `physics/` - Defines SI Arrhenius chemistry, local characteristic/face boundary conditions and state conversions.
  * `geometry/` - Defines orthogonal grid metrics, physical coordinates, component bases and portable spatial ordering.
  * `drivers/` - Provides the inert, reactive, 2D and 3D `inputLoopResult*.m` entry points.
  * `sessions/` - Creates and advances bounded-memory sessions and assembles session data.
  * `io/` - Reads/writes HDF5 checkpoints and exports VTK/XDMF snapshots and time collections.
  * `visualization/` - Contains CSV plotting and 3D slice/profile visualization tools.
  * `validation/` - Contains analytic references, mesh studies and session benchmarks.
  * `gui/interface1D.mlapp` - Contains the original 1D MATLAB App Designer interface.
  * `gui/interface2D.m` / `gui/interface3D.m` / `gui/torchDesktop.m` - Implement the session desktops and their plotting/configuration helpers.
  * `tests/` - Contains MATLAB/Octave numerical tests and MATLAB GUI checks.
* `.github/workflows/python.yml` - Runs Python numerical and GUI tests, Octave checks and cross-language regression tests.
* `.github/workflows/matlab.yml` - Runs MATLAB numerical and GUI tests.

## ⚙️ Installation & Usage

1. **Open the Project:** Download or clone the repository, then navigate to the `TORCH` root directory. Run the following commands from that directory.
2. **Prepare Python:** Create a virtual environment and install the dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run a 1D Simulation:** Execute the Sod benchmark with exact-reference validation, conservation diagnostics and figure exports:

   ```bash
   python3 python/torchCli.py --preset sod_standard --exact_val \
     --out out/sod.csv --json_out out/sod.json --plot out/sod
   ```
4. **Run a 2D Simulation:** Execute a circular pressure perturbation on a Cartesian mesh:

   ```bash
   python3 python/torchCli.py --dim 2d --preset_2d 2d_explosion \
     --nx 40 --ny 40 --t_corte 0.02 \
     --out out/explosion.csv --plot out/explosion
   ```
5. **Run a 3D Simulation:** Execute a spherical pressure perturbation on a Cartesian grid and export a central plane slice:

   ```bash
   python3 python/torchCli.py --dim 3d --preset 3d_explosion \
     --nx 16 --ny 16 --nz 16 --t_corte 0.02 \
     --out out/explosion3d.csv --json_out out/explosion3d.json \
     --plot out/explosion3d --slice_axis z
   ```
6. **Launch the Python Interface:** Open the interactive 1D interface to select a preset, run a simulation and inspect the results:

   ```bash
   python3 python/torchGui.py
   ```
7. **Use MATLAB:** Open MATLAB, navigate to the repository root and add the solver directory to the path. Launch the 1D interface or run a numerical case directly:

   ```matlab
   addpath('matlab'); setupTorch;
   interface1D;

   % Run a 2D case directly
   [U, t, lambda, X, Y, diagnostic] = inputLoopResult2D( ...
       '2d_explosion', 40, 40, 0.02, 0.4, 'VanLeer');

   % Run and plot a 3D case
   result = torchSolver3D('preset_name','3d_explosion', ...
       'nx',16,'ny',16,'nz',16,'t_end',0.02);
   plotTorch3D(result,'z',[],'out/explosion3d_matlab');
   ```
8. **Run Verification:** Install the independent test readers, then execute the Python and MATLAB/Octave numerical suites:

   ```bash
   pip install -r requirements.txt
   python3 -m unittest discover -s python/tests -v
   octave --no-gui --quiet --eval "run('matlab/tests/testMatlabSolver.m'); addpath('matlab/tests'); testMatlabSolver3D;"
   ```

All dimensional inputs use **SI units**, including pressure in **Pa**. Canonical inert Riemann benchmarks retain their conventional numerical values. Detailed reactive examples, solver options, RDE diagnostics and MATLAB GUI checks are included below. Use `python3 python/torchCli.py --help` to list command-line options.

### Python workflows

```bash
# Sod shock tube with exact Riemann validation, audit and figures
python3 python/torchCli.py --preset sod_standard --exact_val \
  --out out/sod.csv --json_out out/sod.json --plot out/sod

# Explicit CLI options override a 1D preset's mesh, time, CFL and limiter
python3 python/torchCli.py --preset strong_shock --n 200 --t_corte 0.012 \
  --out out/strong_shock.csv

# Periodic 1D boundaries
python3 python/torchCli.py --bc Periodic --n 100 --t_corte 0.05 \
  --out out/periodic.csv

# Reactive SI ignition setup
python3 python/torchCli.py --mode reactive --kinetic_model arrhenius \
  --n 100 --t_corte 0.00001 --out out/reactive.csv --json_out out/reactive.json

# 2D explosion with a custom mesh, domain length and end time
python3 python/torchCli.py --dim 2d --preset_2d 2d_explosion \
  --nx 40 --ny 40 --lx 2 --ly 2 --t_corte 0.02 \
  --out out/explosion.csv --json_out out/explosion.json --plot out/explosion

# Reduced annular-channel ignition with fresh-mixture injection
python3 python/torchCli.py --dim 2d --preset_2d 2d_rde_chamber \
  --nx 20 --ny 20 --t_corte 0.0001 --inlet 1 0 100 101325 \
  --ambient_pressure 101325 --out out/rde.csv --json_out out/rde.json

# Mesh refinement: writes measured errors and observed orders to JSON
python3 python/torchCli.py --study convergence --out out/convergence.json

# Standalone figure export; exact overlay is opt-in and assumes standard Sod
python3 python/torchPlot.py --file out/sod.csv --save out/sod_export --exact

# Interactive 1D interface
python3 python/torchGui.py
```

Available 3D presets: `3d_sod_x`, `3d_sod_y`, `3d_sod_z`, `3d_explosion`, `3d_ignition`.

Available 1D presets: `sod_standard`, `rarefaction_wave`, `strong_shock`, `toro_test4`, `detonation_cj`. Available 2D presets: `2d_explosion`, `2d_implosion`, `2d_riemann_quadrant`, `2d_shock_tube`, `2d_rde_chamber`.

`--limiter` accepts `VanLeer`, `Albada`, `MINBEE`, `Superbee`, `Unos`; `--wavespeed` selects the 1D `Pressure Based` or `Enthalpy Based` estimate. `--bc` selects 1D boundaries; 2D CLI boundaries come from the selected preset. Use `python3 python/torchCli.py --help` for all options. CFL must be in `(0, 0.5]`; the CLI defaults to `0.5` in 1D, `0.45` in 2D and `0.4` in 3D.

The Python drivers reject nonfinite or nonphysical initial states, invalid solver options and incomplete custom 2D initialization, including at zero final time. The 1D `divider` argument accepts a fraction in `(0,1)` or a percentage in `(1,100)`; the split is snapped to an interior cell face. Inert `gamma` must be finite and greater than one.

The exact overlay in the simulation CLI and GUI uses the actual initial states and discontinuity location. It is an infinite-domain Riemann reference; comparisons after boundary interactions do not validate a finite-domain solution against that reference. Vacuum-generating Riemann problems are unsupported and raise an error; failure of the pressure iteration also raises an error rather than returning an unverified reference.

### MATLAB / GNU Octave

```matlab
addpath('matlab'); setupTorch;

% Inert Sod shock tube
[U, t] = inputLoopResult(1, 100, ...
    1, 0, 1, 0.125, 0, 0.1, 1.4, 0.4, 0.2, ...
    'VanLeer', 'Sound Speed Controlled', ...
    'Variante 1 - Original', 'Pressure Based', 'Open', 50);

% Reactive SI ignition setup
[U, t, lambda] = inputLoopResultwQR(1, 100, ...
    1, 0, 1e6, 0.5, 0, 101325, ones(1,100), 0.4, 1e-5, ...
    'VanLeer', 'Sound Speed Controlled', ...
    'Variante 1 - Original', 'Pressure Based', 'Open', 50);

% 2D case, including inlet/outlet mass balance and final exit thrust
[U, t, lambda, X, Y, diagnostic] = inputLoopResult2D( ...
    '2d_rde_chamber', 20, 20, 1e-4, 0.35, 'VanLeer');
```

Run `addpath('matlab'); setupTorch;` from the repository root before using the MATLAB functions or opening `matlab/gui/interface1D.mlapp` in App Designer. If the working directory is elsewhere, pass the absolute path to the `matlab` directory to `addpath`. The desktop interfaces have been exercised in graphical sessions; repeatable checks are provided in [the MATLAB GUI tests](matlab/tests/testMatlabGui.m). Tk regressions run with the Python suite when a display is accessible and skip otherwise. Run the MATLAB GUI regressions separately:

```matlab
addpath('matlab/tests');
testMatlabGui;
```

## 🔬 Numerical Methods & Physical Model

### Implemented methods

- Conservative Euler updates with MUSCL-Hancock reconstruction and HLLC interface fluxes. The 2D predictor includes both directional flux gradients and transports tangential momentum.
- Van Leer, Van Albada, MINBEE, Superbee and bounded Unos slope options.
- Open, reflective and periodic boundaries in both dimensions.
- An Arrhenius reaction model shared between the 1D, 2D and 3D implementations of each language. Python also provides a reduced induction/exothermic two-stage model through `--kinetic_model znd`.
- Composition-dependent `gamma` throughout reactive pressure conversion, CFL calculation, hydrodynamics and final output.
- Conservative upwind transport of `rho*lambda` and, for Python's two-stage model, `rho*beta`, using the hydrodynamic mass flux.
- Symmetric reaction/hydrodynamic/reaction splitting with exponential-midpoint chemistry. CFL is checked again after the first chemical half-step because heat release changes sound speed.
- A conservative first-order local Lax–Friedrichs fallback when a hydrodynamic update would produce nonpositive density or internal energy. Integration errors and step limits raise errors rather than returning a successful partial simulation.

Hydrodynamic reconstruction is second order for smooth solutions. Species transport is first order; symmetric splitting alone does **not** establish second-order accuracy of the complete reactive calculation. Discontinuities and fallback steps also reduce observed accuracy.

### Units and reduced physical model

All dimensional input uses **SI units**:

| Quantity                      | Unit   |
| ----------------------------- | ------ |
| Density `rho`               | kg/m³ |
| Velocity `u`, `v`, `w`         | m/s    |
| Pressure `p`                | Pa     |
| Total energy per volume `E` | J/m³  |
| Length and time               | m, s   |
| Heat-release parameter `Q`  | J/kg   |

Pressure units are never inferred from numerical magnitude. To migrate an older reactive setup expressed in atmospheres, multiply its input pressures by `101325` **before initializing conservative states**. Canonical inert Riemann benchmarks retain their conventional numerical values, such as Sod's `pL=1`, `pR=0.1`; no atmospheric conversion is applied.

The Euler state is `[rho, rho*u, E]` in 1D, `[rho, rho*u, rho*v, E]` in 2D and `[rho, rho*u, rho*v, rho*w, E]` in 3D, with

```text
E = p/(gamma-1) + rho*(u²+v²+w²)/2
```

`lambda=1` represents fresh reactants and `lambda=0` burned gas. The reduced mixture closure is

```text
gamma(lambda) = (4.5-lambda)/(3.5-lambda)
M(lambda) = (2/3)*lambda*2.016 + (1/3)*lambda*32 + (1-lambda)*18.016  [kg/kmol]
T = p*M/(rho*8314.46)                                                 [K]
rate = 1.76e5*exp(-(17.15*4184)/(8.314*T))                            [1/s]
delta_E = rho*141584000*(lambda_old-lambda_new)                       [J/m³]
```

The temperature equation uses the gas constant in J/(kmol·K); Arrhenius activation energy is in J/mol and uses the gas constant in J/(mol·K). These are different numerical constants with consistent units.

The chemistry and mixture closure are reduced models. Quantitative detonation and RDE performance predictions require additional physical calibration and validation; the historical `detonation_cj` preset name is retained for compatibility and does not certify a Chapman–Jouguet speed. The drivers solve Euler equations on fixed grids; `torchAnalysis.py` also provides standalone viscous-flux helpers and refinement indicators.

### RDE boundary model and diagnostics

The reduced RDE case has periodic boundaries in `x`, a fixed fresh-mixture inlet at `y=0`, and a transmissive outlet at `y=Ly`. Preset patterns are defined in normalized cell-center coordinates and scale with the requested domain size.

The default inlet primitive state is `[rho,u,v,p] = [1,0,100,101325]` in SI, with `lambda=1` and, where applicable, `beta=1`. Python accepts `inlet_primitive` in `run_torch_solver_2d`; the CLI exposes it as `--inlet RHO U V P`. MATLAB accepts the inlet vector as its eighth preset argument or fourteenth custom argument. This is a prescribed inlet, not a pressure-responsive injector or nozzle model.

Python 2D results and JSON output include `exit_thrust_per_depth`:

```text
integral_at_exit(rho*v² + p - p_ambient) dx
```

This is thrust per unit out-of-plane depth in N/m. Python exposes `ambient_pressure`; MATLAB accepts ambient pressure as its ninth preset argument or fifteenth custom argument. The default is `101325 Pa`. The thrust diagnostic assumes the top edge is an exit.

`mass_change_rel` measures the change of mass inside the domain. `mass_conservation_error` in 2D accounts for integrated inlet/outlet flux, so physical outflow is not reported as numerical mass loss. The 1D CLI audit also accounts for boundary flux and chemical energy release. Calling `compute_conservation_audit` without boundary integrals returns changes, with conservation residuals set to `None` because the boundary exchange is unknown.

### Cartesian 3D model, usage and diagnostics

The 3D solvers use **Cartesian uniform grids** and retain only the final state. Conservative and primitive arrays have shape `(5,nz,ny,nx)` in Python and `[5,nz,ny,nx]` in MATLAB. Coordinates `X`, `Y`, `Z` and composition arrays have shape `(nz,ny,nx)`. The primitive ordering is `[rho,u,v,w,p]`. Arrays and spacings are intentionally distinguished: lengths and spacings are supplied in **x,y,z** order.

`run_torch_solver_3d` and `torchSolver3D` accept the same named mesh, domain, limiter and boundary options. MATLAB uses name/value arguments. Custom initialization uses `primitive_init`; it overrides the preset. Set `is_reactive=True` (Python) or `true` (MATLAB) for a custom reactive case. MATLAB implements Arrhenius kinetics; Python additionally supports `kinetic_model='znd'` and `beta_init`. `lambda_init` and `beta_init` must contain one bounded fraction per cell. Presets use normalized cell centers, so they scale with `lx`, `ly` and `lz`.

The 3D defaults are 20 cells per axis and CFL `0.4`; the final time defaults to `0.05 s`, or `1e-5 s` for the ignition preset. Each axis supports open, reflective and periodic boundaries. These legacy drivers remain Cartesian. The v1.2 session API adds prescribed/reservoir inlets, pressure outlets and curved grids in 3D; these are generic Euler boundary models, rather than a calibrated 3D RDE injector. The existing 2D RDE and thrust diagnostic retain their scope.

The unsplit predictor includes all three flux gradients. The time step satisfies

```text
dt = CFL / (max(|u|+a)/dx + max(|v|+a)/dy + max(|w|+a)/dz)
```

Chemistry uses the existing SI model and symmetric reaction/hydrodynamic/reaction splitting, with a renewed CFL check after heating. First-order upwind species transport retains its existing accuracy limits. Invalid reconstructed states or a nonphysical update trigger a conservative first-order local Lax–Friedrichs step. Integration stalls and step limits raise errors. The 3D bounded Unos limiter includes both reflected MUSCL bounds to preserve reflection symmetry; the existing 1D/2D limiter choices are retained.

The returned `initial_integral`, `final_integral`, `boundary_integral`, `source_integral` and `balance_residual` contain five entries, ordered as mass, three momenta and energy. Face fluxes are integrated outward over all six faces; chemical heat release contributes to the energy source. The residual is `final - initial + boundary - source`. In 3D these are volume integrals: mass is in kg and energy in J. The final state uses memory proportional to the cell count; reconstruction and flux work arrays also consume memory, so refine all three directions with care. No time-history snapshots are stored by default.

```bash
# 3D reactive SI ignition, with per-axis boundaries
python3 python/torchCli.py --dim 3d --preset 3d_ignition \
  --nx 12 --ny 8 --nz 8 --t_corte 1e-5 --bc_x Open --bc_y Periodic --bc_z Periodic \
  --out out/ignition3d.csv --json_out out/ignition3d.json --plot out/ignition3d

# Plot an existing volume on a plane normal to x
python3 python/torchPlot.py --file out/ignition3d.csv --save out/ignition3d_x \
  --slice_axis x --slice_index 5
```

CSV exports include `x,y,z,p,rho,u,v,w` and reactive `lambda`/`beta` where applicable. JSON includes the five-equation balances and mesh metadata. Figure exports contain a plane slice and its centerline profiles; a complete rectangular CSV grid is required, but row order is arbitrary. Python slice indices are **zero-based**, MATLAB indices are **one-based**; omission selects the middle plane. Python 3D PNG exports use 200 DPI and MATLAB uses 200 DPI; both also export SVG. Existing 1D/2D Python plots retain their 600-DPI export.

```matlab
% Name/value wrapper returning the final state and diagnostics
[U,t,lambda,X,Y,Z,diagnostic] = inputLoopResult3D( ...
    'preset_name','3d_ignition','nx',12,'ny',8,'nz',8,'t_end',1e-5);

% Numerical 3D regressions (MATLAB or Octave)
addpath('matlab/tests');
testMatlabSolver3D;
```

## 💾 Checkpoints, Restart & Temporal Output

The **v1.1** session APIs add checkpoint/restart and scientific output in all three dimensions. Existing numerical drivers, return values and GUI workflows remain available. The legacy 1D drivers still return the full history; use sessions when only the current state should remain in memory. The v1.2 extensions below add orthogonal curved grids and face conditions to sessions. The v1.3 desktops expose these sessions graphically; adaptive meshes remain future work.

### Python CLI

```bash
# Run two complete steps, write a restart checkpoint and a temporal series
python3 python/torchCli.py --preset sod_standard --n 100 --t_corte 0.2 \
  --stop_after_steps 2 --checkpoint out/sod_restart.h5 --checkpoint_every 1 \
  --series out/sod_history --output_every 5 --out out/sod_partial.csv

# Continue using the saved mesh, physics, numerical options and target time
python3 python/torchCli.py --restart out/sod_restart.h5 \
  --checkpoint out/sod_restart.h5 --checkpoint_every 5 \
  --series out/sod_history --output_every 5 \
  --vtk out/sod_final.vtr --xdmf out/sod_final.xdmf \
  --out out/sod_final.csv --json_out out/sod_final.json

# The same output options apply to 2D and 3D
python3 python/torchCli.py --dim 3d --preset 3d_explosion \
  --nx 12 --ny 10 --nz 8 --t_corte 0.02 \
  --checkpoint out/explosion_restart.h5 --checkpoint_every 2 \
  --series out/explosion_history --output_every 2 \
  --vtk out/explosion_final.vtr --xdmf out/explosion_final.xdmf \
  --out out/explosion_final.csv
```

`--stop_after_steps N` pauses after **N additional completed steps** and explicitly reports `paused`. Omitting it runs to the configured end time or raises on an integration/step-limit error. Checkpoints are written every `--checkpoint_every N` steps and at the final or paused state. An interval requires a destination. `--series` stores the initial/resumed state, every `--output_every N` completed steps and the final state. Formats can be restricted with `--output_formats vtk` or `--output_formats xdmf`.

On restart, the checkpoint supplies the dimension, state, mesh, boundary conditions, composition, numerical parameters, counters and conservation integrals. Configuration overrides are rejected except for an explicitly supplied `--t_corte`, which changes the target end time. CSV, session JSON and figure outputs remain available. Session JSON reports `completed`, the full configuration and integrated balances. Session IO is separate from the legacy `--study` and `--exact_val` workflows; combining these options is rejected.

### Python session API

```python
from cfdSolver import Simulation
from cfdSolver.torchIO import write_vtk, write_xdmf

simulation = Simulation.preset("3d_ignition", [12, 8, 8], t_end=1e-4)
simulation.run(step_limit=2, checkpoint="out/ignition.h5")

resumed = Simulation.load("out/ignition.h5")
resumed.run(checkpoint="out/ignition.h5", checkpoint_every=5,
            output_prefix="out/ignition_history", output_every=5)
write_vtk("out/ignition.vtr", resumed)
write_xdmf("out/ignition.xdmf", resumed)
```

Run this example with `PYTHONPATH=python`, or add the repository’s `python` directory to your import path. `Simulation(primitive, lengths, t_end=..., ...)` accepts custom states, with primitive arrays shaped `(dimension+2, ...spatial axes in reverse order)` and lengths/counts/boundaries in **x,y,z** order. It supports the existing inert, Arrhenius and Python ZND physics; `lambda_init` and `beta_init` use the spatial shape. `state`, `lam`, `beta`, `time`, `steps` and integrated balances expose the current result. A session stores no list of solution snapshots.

### MATLAB session API

```matlab
addpath('matlab'); setupTorch;

% Custom 2D primitives [rho,u,v,p], with SI pressures
P = ones(4,8,12);
P(2:3,:,:) = 0;
P(4,:,:) = 1e6;
simulation = torchCreateSession(P,[1,1], ...
    't_end',1e-4,'reactive',true,'boundaries',{'Periodic','Periodic'});
simulation = torchRunSession(simulation,'step_limit',2, ...
    'checkpoint','out/ignition_matlab.h5');

resumed = torchReadCheckpoint('out/ignition_matlab.h5');
resumed = torchRunSession(resumed,'checkpoint','out/ignition_matlab.h5', ...
    'checkpoint_every',5,'output_prefix','out/ignition_matlab_history','output_every',5);
torchWriteVTK('out/ignition_matlab.vtr',resumed);
torchWriteXDMF('out/ignition_matlab.xdmf',resumed);
```

MATLAB primitive layouts are `[3,nx]`, `[4,ny,nx]` and `[5,nz,ny,nx]`. The session structure contains the same state and balance quantities as Python. `torchReadCheckpoint(path,'t_end',...)` can explicitly extend the target time. Inert and Arrhenius checkpoints can be continued in either language; **reactive ZND restart in MATLAB is rejected** because that kinetic model remains Python-only. Checkpoint/scientific IO requires MATLAB’s native HDF5 functions; GNU Octave remains supported for the numerical solvers and numerical regressions, but this release does not supply an Octave HDF5 IO implementation.

### Format and restart guarantees

New files use schema **2**, solver family **1.4**. Readers also accept schema **2** / family **1.2** checkpoints from v1.2/v1.3 and original schema **1** / family **1.1** Cartesian checkpoints. Loading preserves their saved state, configuration and balances; saving upgrades the family to 1.4. Continued 1D HLLC simulations use the consistency correction below, so cross-version continuation need not reproduce the older solver's trajectory. Older readers reject family 1.4 explicitly. The format stores a UTF-8 JSON byte vector at `/metadata` and separate float64 conservative components and species under `/state`. Components use **x-fastest** flat ordering, independent of Python/MATLAB array storage. Metadata records geometry, coordinate origins, face conditions (when used), SI units, dimension, grid configuration, run identity, target/current times, step counter, initial totals and accumulated boundary/source integrals. It also stores current totals for a consistency check. No executable object serialization or pickle is used.

Readers reject unknown schema/solver versions, unsupported geometries or kinetics, missing or incompatible datasets, nonphysical states and invalid counters/balances. Schema 2 supports the uniform logical grids in the geometry table below; future mesh formats require an explicit schema migration. Writers close a complete temporary file in the destination directory before replacing the checkpoint. A failed write preserves the previous valid file; this is not a substitute for backups or a guarantee against storage/power failure.

For the **same executable environment and unchanged target/configuration**, restart at a completed step boundary reproduces the uninterrupted step sequence and results exactly. Cross-language comparisons use floating-point tolerances. Saving output does not shorten time steps. Forcing an earlier end time and later extending it can change the step partition, so it is not the equivalence guarantee used in the tests.

Temporal collections use a run identity to prevent accidental reuse by another simulation. Resuming at the last recorded frame appends without duplicating it. If the series is ahead of an older checkpoint, choose a new prefix; old frames are not silently removed. Frame data are written before collection indexes, and a `.series.json` manifest tracks only metadata. Memory use is proportional to the current mesh plus index metadata; disk use grows with the number of stored frames. Moving a collection requires keeping its relative frame/HDF5 paths together.

### Scientific outputs

| Output | Contents |
| --- | --- |
| `.vtr` | VTK XML rectilinear grid with cell-centered conservative variables, pressure, active velocities and species |
| `.vts` | VTK XML structured grid with physical Cartesian vertices for curved meshes |
| `.pvd` | VTK time collection referencing individual `.vtr` or `.vts` frames |
| `.xdmf` + `.h5` | XDMF geometry/connectivity and cell-centered HDF5 fields |
| temporal `.xdmf` | XDMF time collection referencing individual HDF5 frames |
| `.series.json` | Run/configuration identity and frame metadata for safe continuation |

XDMF uses line, quadrilateral or hexahedral cells in 1D, 2D or 3D. Static snapshot time remains in HDF5 metadata; temporal collections expose time for each grid. Visualization HDF5 snapshots also contain the restart schema and can be loaded as checkpoints. Export coordinates are **cell vertices**; solution variables are **cell-centered**. Inactive VTK axes have a single coordinate. VTK XML is ASCII, while XDMF field arrays are float64 HDF5 datasets. Curved grids use VTK StructuredGrid (`.vts`); Cartesian grids use RectilinearGrid (`.vtr`). PVD collections select the correct frame extension.

Validation uses independent **VTK** and **meshio** readers to recover geometry, connectivity, values and time collections from files produced by both languages. These validation tools are included alongside the solver dependencies in the unified `requirements.txt`. The formats target viewers such as ParaView; interactive viewer testing is not part of automated CI.

## 🌐 Geometries & Face Boundary Conditions

The **v1.2** extensions use `Simulation` / `torchCreateSession` and the checkpoint-capable CLI. The legacy solver/driver APIs remain Cartesian. Logical coordinates are uniform; metric factors supply exact cell volumes and face areas. Conservative and primitive arrays keep the existing reversed spatial layout; velocity/momentum components follow the **local orthonormal basis** in this table.

| Geometry | 1D | 2D | 3D | Domain and integral convention |
| :--- | :--- | :--- | :--- | :--- |
| `cartesian` | x | x, y | x, y, z | Length, area per unit depth, or volume |
| `cylindrical` | r | r, z (axisymmetric, no swirl component) | r, theta, z | 1D: full circumference per unit axial length; 2D: full 2π revolution; 3D: specified angular extent |
| `annular` | r | r, theta | r, theta, z | Positive inner radius; 1D: full circumference per unit axial length; 2D: sector per unit depth; 3D: specified sector volume |
| `spherical` | r | r, theta (axisymmetric, no azimuthal velocity) | r, theta, phi | 1D: full 4π solid angle; 2D: full 2π azimuth; 3D: specified angular extent |

`lengths` / `lengths_xyz` specify coordinate **extents**, and `origins` / `origins_xyz` specify lower coordinates. The names `x/y/z` in mesh configuration and face keys denote the first/second/third **logical axes**. Radial/axial values are in metres; angles are in radians; spherical theta is colatitude in `[0,π]`. Azimuthal extents cannot exceed `2π`, and each azimuthal cell spans less than `π`. Periodic azimuth requires a full `2π` extent. Radial and spherical-polar coordinates cannot be periodic.

Reduced radial and axisymmetric models support `r=0` with a reflective normal condition. Spherical 2D supports both poles with reflective theta conditions. With default open axis options, sessions select these symmetry faces automatically; explicitly conflicting conditions are rejected. **3D cylindrical grids require positive inner radius; 3D spherical grids require positive inner radius and `0<theta<π`.** These shell/wedge domains avoid singular connectivity across an axis or pole. Full 3D solid cylinders/spheres across those singularities are outside this release. A Cartesian spherical-explosion preset continues to describe an initial pressure shape on a rectangular mesh.

### Numerical formulation and balances

Cartesian sessions without face overrides retain the existing MUSCL–Hancock/HLLC kernel. Curved grids and Cartesian sessions with `face_boundaries` use **limited primitive MUSCL reconstruction, Rusanov fluxes and SSPRK2**. All five existing limiters are supported. Invalid reconstructed/stage states trigger a first-order fallback; a failed positive update halves the complete split step and recomputes chemistry. Legacy HLLC variant/controller/wavespeed switches are rejected when they would select an unsupported option in this kernel. Chemistry retains reaction/2–hydrodynamics–reaction/2 splitting, and species transport remains conservative first order.

Each update integrates face fluxes with physical areas and divides by physical cell volume. Cylindrical/spherical momentum equations include geometric pressure and inertial sources. The discrete pressure source balances a uniform fluid at rest, including cells adjacent to supported symmetry axes. `initial_integral`, `integral`/`torchSessionIntegral`, `boundary_integral` and `source_integral` use the table's physical measure. The audit remains `final - initial + boundary - source`; **source now includes geometric momentum terms as well as reaction energy**. Momentum components are local-basis integrals, not a global Cartesian momentum vector. This scheme does not promise exact discrete angular-momentum conservation; rotating-equilibrium and radial-expansion tests measure convergence. Mass and total energy have no geometric source.

The metric formulation can be compared with [Clawpack's axisymmetric Euler equations](https://www.clawpack.org/gallery/pyclaw/gallery/shock_bubble_interaction.html) and [the Athena++ coordinate-source discussion](https://doi.org/10.3847/1538-4365/ab929b). TORCH's implementation and validation cases are independent; these references describe the underlying coordinate equations.

### Per-face conditions

Pass `face_boundaries` as a dictionary/struct keyed by `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`. Unspecified faces inherit the axis options and geometry symmetry defaults. Checkpoints store a normalized list in that order. Supported types are:

| Type | Data | Behavior |
| :--- | :--- | :--- |
| `wall` | None | Slip wall: reflected normal velocity, unchanged tangential components; zero mass and energy flux |
| `outflow` | None | Zero-gradient extrapolation; this is not a nonreflecting acoustic condition |
| `periodic` | Both faces on the axis | Wraps states, reconstruction and species |
| `inflow` | `primitive`: density, local velocities, pressure | Prescribed ghost state applied through a Riemann flux; normal velocity must point inward |
| `pressure_outlet` | Positive `pressure`; optional `backflow` primitive state | Subsonic isentropic pressure state preserving the outgoing normal invariant; supersonic outflow extrapolates without imposing pressure; inward flow requires explicit backflow data |
| `stagnation_inlet` | `total_pressure`, `total_temperature`, optional `gas_constant` (default `287`) | Normal reservoir inflow from the outgoing interior invariant and total conditions; sonic choking caps Mach at one; incompatible reverse supply raises an error |

Inflow/reservoir/backflow data may include `lambda` and `beta` in `[0,1]` (default fresh mixture: `1`). Reactive incoming gamma follows lambda; outgoing fractions extrapolate. Primitive pressure is static pressure in Pa; reservoir temperature is K and gas constant is J/(kg·K). Select a gas constant appropriate to the modeled mixture. Reservoir inflow has zero tangential velocity; use prescribed primitives for oblique injection. These inviscid ghost-state/characteristic approximations do not model injector losses, turbulence or plenum dynamics. Prescribing all inlet primitives is primarily suitable for supersonic inflow; subsonic reservoir coupling uses `stagnation_inlet`.

### Examples and output coordinates

```python
import numpy as np
from cfdSolver import Simulation

# Annular rotating flow, local components [rho, u_r, u_theta, p].
r = 0.5 + (np.arange(12) + 0.5) * 0.5 / 12
P = np.ones((4, 24, 12))
P[1] = 0.0
P[2] = 20.0 * r
P[3] = 101325.0 + 0.5 * (20.0 * r)**2
simulation = Simulation(P, [0.5, 2*np.pi], origins=[0.5, 0.0],
    geometry="annular", t_end=1e-4, face_boundaries={
        "x_min": "wall", "x_max": "wall",
        "y_min": "periodic", "y_max": "periodic"})
simulation.run(checkpoint="out/annulus.h5", output_prefix="out/annulus")
```

```matlab
addpath('matlab'); setupTorch;
% Spherically symmetric radial blast, including the origin.
P = [ones(1,80); zeros(1,80); 101325*ones(1,80)];
P(3,1:20) = 1e6;
s = torchCreateSession(P,1,'geometry','spherical','t_end',1e-4);
s = torchRunSession(s,'checkpoint','out/radial.h5', ...
    'output_prefix','out/radial','output_every',5);
```

```bash
# Uniform annular sector, scientific output in physical coordinates.
python3 python/torchCli.py --dim 2d --geometry annular --origins 0.5 0 \
  --nx 24 --ny 32 --lx 0.5 --ly 6.283185307179586 \
  --primitive 1 0 0 101325 --t_corte 1e-4 \
  --checkpoint out/annular.h5 --vtk out/annular.vts --out out/annular.csv
```

For custom face conditions, pass a JSON object using the table above with `--face_boundaries path/to/faces.json`. `--primitive` initializes a uniform local primitive state and excludes explicit presets. `--lambda_value` / `--beta_value` initialize uniform composition for reactive sessions. Existing presets can initialize arrays in logical coordinates; their physical interpretation must match the selected geometry. On restart, geometry, origins, boundary data and composition come from the checkpoint; overrides are rejected except for target end time.

Checkpoint component names follow the local coordinates (`momentum_r`, `momentum_theta`, etc.). Cartesian names remain unchanged. CSV uses logical coordinates and local velocity labels (`r,theta,p,rho,u_r,u_theta`, for example). VTK/XDMF vertices are mapped to physical Cartesian positions, and curved-grid files include local velocities plus `velocity_cartesian_x/y/z`. A 1D output is a radial ray; axisymmetric 2D is a meridional section, rather than a generated revolved volume. The metadata and metric integrals retain the symmetry factors above. Curved-grid visualization uses VTK/XDMF; the legacy CSV `--plot` path remains Cartesian and rejects curved grids before integration.

## 🖥️ 2D / 3D Session Desktops

The **v1.3** desktops use the same session APIs and checkpoints as the CLI. Launch from the repository root:

```bash
python3 python/torchGui.py                  # Original 1D interface
python3 python/torchGui.py --dim 2d         # Session desktop, 2D
python3 python/torchGui.py --dim 3d         # Session desktop, 3D
python3 python/torchDesktop.py --dim 3d     # Equivalent dedicated launcher
```

```matlab
addpath('matlab'); setupTorch;
interface1D;                              % Original App Designer app
app2 = interface2D;                        % Session desktop, 2D
app3 = interface3D;                        % Session desktop, 3D
```

MATLAB desktops require MATLAB with `uifigure`, native HDF5 and a graphical display; no Parallel Computing Toolbox is needed. Octave supports their numerical models and slice checks, but the windows require MATLAB. Python requires Tkinter and a graphical or virtual display.

### Configure and continue

* **Setup:** Select dimension, geometry, preset, end time, CFL, limiter and chemistry. Reactive gamma follows composition; inert gamma is configurable. Uniform/geometric initial conditions expose density, local velocities, static pressure and rotation rate where applicable. Python supports Arrhenius/ZND; MATLAB supports Arrhenius.
* **Mesh:** Enter integer cell counts, coordinate extents and origins. Labels distinguish metres from radians and follow the geometry. Defaults select valid shell/wedge domains; v1.2 singularity restrictions still apply.
* **Boundaries:** Select a face and condition, enter its active physical parameters and press **Apply face condition**. Periodic selection applies to both faces. Changing one of a periodic pair to another condition changes the other to outflow, which can then be edited separately. Inlet/backflow velocities must point inward. Outlet pressure and backflow static pressure share the editor's pressure value.
* **Files:** Choose a results folder and recording interval, or load a restart/recorded series. **Run / resume** starts or continues the latest state. **Pause** stops between complete steps. **New** clears the active simulation and unlocks setup; written results remain on disk.

Mesh, physics and boundary controls are locked while a session is active. After pausing/loading, the saved configuration remains authoritative; the target end time may be extended. Choose **New** to configure different physics or mesh. A completed simulation needs a later end time to resume. Python gives mutable state to one worker and keeps Tk operations on the UI thread. MATLAB advances cooperatively with a step timer. Large individual steps or writes may delay a requested pause; neither implementation interrupts a partial numerical update.

Cartesian initial conditions include existing 2D/3D benchmarks. Curved grids offer **Uniform**, **Radial blast**, and **Rotating equilibrium** where the local velocity components permit it. The blast multiplies pressure by ten in the inner quarter of the radial extent. Rotation initializes `Omega*r` (or `Omega*r*sin(theta)` in spherical 3D) and its quadratic pressure profile. These are initial conditions for the existing coordinate equations, rather than new injector/turbulence models.

### Explore, record and export

Choose pressure, density, speed, a local velocity, unburned fraction or Python ZND induction fraction. A 2D **Map** uses physical Cartesian vertices on curved grids. A 3D **Map** shows the selected pair of logical coordinates; **3D plane** renders actual Cartesian slice vertices, including curved constant-radius surfaces. The lower plot shows the middle-row profile along its horizontal logical coordinate. Slice indices are **zero-based in Python** and **one-based in MATLAB**; new 3D views open centrally. Matplotlib's toolbar and MATLAB axes tools provide zoom, pan and 3D navigation.

Every new simulation/checkpoint branch gets a fresh run subfolder. Initial, periodic and final/paused frames are restartable HDF5/XDMF snapshots. **Open recorded series** reads a `.series.json` collection; the timeline, **Play** and **Latest** load one frame at a time. Playback requires HDF5/XDMF frames, so VTK-only collections cannot be replayed as restarts. Compatible collections from either language can be reopened and continued. Recording leaves the numerical step sequence unchanged.

RAM contains the current simulation and displayed frame, index metadata and solver work arrays; there is no full solution history in memory. Disk use grows with mesh size and recording frequency. Recorded files remain after closing. Loading a checkpoint starts a fresh collection when resumed; reopening a series continues its latest recorded state.

**Save current restart** saves the latest paused/completed simulation, even when viewing an earlier frame. **Export displayed data** writes that viewed frame as CSV, HDF5, VTK or XDMF. **Export displayed figure** writes PNG/SVG. Resume always uses the latest simulation state. Displayed time, step and balance labels identify the visible frame; CSV keeps the v1.2 coordinate/local-component conventions.

The desktops retain the supported geometry domains and use the current session/checkpoint implementation. Tests cover real controls, workers/timers, pause/resume, configuration locking, physical slices, playback, exports, invalid inputs and closing during a run. Cross-desktop checks continue Python checkpoints in MATLAB and then resume their recorded series in Python.

## 📏 Independent Analytic Studies & Benchmarks

The **v1.4** studies compare conservative cell averages with references defined independently of TORCH's flux, reconstruction and Riemann routines. They exercise both the legacy **HLLC** and explicit-face **MUSCL–Rusanov/SSPRK2** session kernels:

* **Isentropic vortex:** A strength-5 translating vortex on a periodic `10 × 10` domain, in 2D and a four-cell 3D extrusion with axial velocity `0.25`. Initial/reference conservative averages use two-point Gauss quadrature in each varying direction. The analytic profile follows the [NASA vortex study](https://ntrs.nasa.gov/citations/20150018403). The periodic domain truncates exponentially small vortex tails; this study measures practical second-order behaviour, rather than arbitrary-order accuracy.
* **Oblique acoustics:** A diagonal, right-running linear Euler mode in 1D/2D/3D, with `rho0 = p0 = 1`, `gamma = 1.4`, background velocity `0.2` in each direction and relative amplitude `1e-6`. Its cell-average sine factor is integrated analytically. Propagation speed in the summed-coordinate phase is `0.2*dimension + sqrt(1.4*dimension)`. The nonlinear Euler/reference difference is of order amplitude squared, so the reference has a finite accuracy floor.
* **Stationary normal shock:** A Mach-2 shock at `x = 0.5`, with 2D/3D transverse extrusions. Upstream `rho = p = 1`, downstream `rho = 8/3`, `p = 4.5`, and velocity follows mass conservation. These states come from [NASA Glenn's normal-shock relations](https://www.grc.nasa.gov/www/k-12/airplane/normal.html). The face kernel uses prescribed inflow and downstream pressure; the HLLC case uses open boundaries. Reports track density error, plateau error, shock position and conservation.

Run full or smaller CI studies:

```bash
python3 python/torchVerify.py --out out/verification-python.json
python3 python/torchVerify.py --quick --out out/verification-python-quick.json
```

```matlab
addpath('matlab'); setupTorch;
report = torchVerify('output', 'out/verification-matlab.json');
assert(report.passed);
report = torchVerify('quick', true, 'output', 'out/verification-matlab-quick.json');
assert(report.passed);
```

Both languages provide 16 case/dimension/kernel studies: 48 runs in the full suite or 32 with `--quick` / `'quick', true`. JSON records mesh counts, time, steps, errors, balance residuals, observed density orders and pass/fail status. Refinement must give density order above `1.0` for smooth cases and `0.6` for shocks; full 1D acoustic studies require order above `1.6`. Errors below `5e-13` are treated as roundoff and their order is `null`. Relative balance residuals must stay below `2e-10`; shocks also check plateau density error below `0.03` and position error below two coarse cells. Python exits unsuccessfully on a failed study; MATLAB/Octave return `report.passed` for the caller to assert. These are analytic code-verification checks; physical RDE calibration retains the limits stated above.

### 1D consistency correction

The new acoustic test exposed two issues in the earlier 1D MUSCL/HLLC implementation. A limiter normalization reduced even a linear gradient, and the HLLC contact speed used cell-average states instead of the MUSCL-predicted interface states. Both languages now reproduce a linear slope and compute the contact wave from the interface momentum balance. A dedicated 24/48/96-cell acoustic regression requires observed density order above `1.7`. Checkpoint schema remains 2; solver family 1.4 identifies this numerical change. Restart within the current version still reproduces continuous integration exactly.

### Measure time and work storage

HLLC now processes at most **8192 faces per batch**, preserving each directional flux while limiting its scratch arrays. Reconstruction creates only the used faces and releases unused predictor temporaries. The Python metric kernel uses adjacent array views and builds only boundary planes for its CFL calculation. MATLAB caches **one** grid keyed by dimension, geometry, counts, extents, origins and periodic axes; `clear torchSessionGrid` releases it. Metric indices use `uint32` when they fit, otherwise `uint64`. Remaining solution, slope and flux arrays still scale with the number of cells; these are fixed-grid CPU solvers.

```bash
python3 python/torchBenchmark.py --quick --out out/benchmark-python.json
python3 python/torchBenchmark.py --steps 10 --repeats 5 --out out/benchmark-python-full.json
```

```matlab
addpath('matlab'); setupTorch;
torchBenchmark('quick', true, 'output', 'out/benchmark-matlab.json');
torchBenchmark('steps', 10, 'repeats', 5, 'output', 'out/benchmark-matlab-full.json');
```

Benchmarks cover 1D/2D/3D Cartesian, spherical 3D and reactive annular 3D sessions. They exclude GUI rendering and scientific-output writes. Timings are the median per-step wall time of repeated runs **after warmup**, with tracing disabled. JSON includes sample times, runtime versions, case definitions and conservation residuals; Python also records solver-source hashes. MATLAB additionally measures repeated one-step calls, representing desktop stepping.

Python reports peak/retained **traced allocations during measured steps**, after the initial state and geometry have been created; NumPy allocations are included where supported by its allocator. This is neither RSS nor the total RAM needed by a simulation. MATLAB reports the structural byte count of a metric grid using `whos`, including indices; this is neither a peak-memory measurement nor directly comparable to the Python allocation metric. No wall-time threshold gates CI because runner load varies. Numerical equivalence, bounded HLLC batch size, lower large-face allocation, bounded session history and cache invalidation have explicit regression tests.

To compare Python solvers from another checkout/archive, run the **current** benchmark tool with `--source-root path/to/checkout --label v1.3.0`, then run it against the current tree with the same mesh/count options. Use fresh processes and the same interpreter, dependencies and hardware, and avoid competing workloads. The 1D correction changes its solution trajectory; performance-only equivalence comparisons apply to unchanged 2D/3D and metric kernels.

### Local v1.3 / v1.4 comparison

Measured on an Intel Core i7-9700KF under Linux, Python 3.12.3 / NumPy 1.26.4 and MATLAB R2024a, using the full benchmark meshes, two warmup steps, five measured steps and three repetitions. Python used `OPENBLAS_NUM_THREADS=1`. These are local observations, not portable speed guarantees; the JSON reports are attached to the [v1.4.0 release](https://github.com/BFG508/TORCH/releases/tag/v1.4.0).

| Python case | v1.3 ms/step | v1.4 ms/step | v1.3 traced peak MiB | v1.4 traced peak MiB |
| --- | ---: | ---: | ---: | ---: |
| Cartesian 2D, 192² | 72.67 | 24.71 | 31.15 | 17.32 |
| Cartesian 3D, 40³ | 93.76 | 83.39 | 77.25 | 35.98 |
| Spherical 3D, 40³ | 117.60 | 123.41 | 56.54 | 40.83 |
| Reactive annular 3D, 24³ | 20.24 | 16.78 | 14.17 | 11.13 |

The Python spherical case uses less work storage with a small time increase on this machine. MATLAB's 192² Cartesian case measured **39.27 → 28.07 ms/step**; its 40³ spherical case measured **154.79 → 140.10 ms/step**, while the structural metric-grid size fell from **13.07 → 8.64 MiB**. MATLAB's 40³ Cartesian timing was effectively unchanged. The different Python/MATLAB memory definitions above apply to these values.

## ✅ Verification

All Python sources, including tests and command-line tools, follow PEP 8.
`setup.cfg` sets a 79-character code limit and a 72-character comment/docstring
limit. The Python workflow checks this with the unified requirements:

```bash
python3 -m pycodestyle python
```

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s python/tests -v
octave --no-gui --quiet --eval "run('matlab/tests/testMatlabSolver.m'); addpath('matlab/tests'); testMatlabSolver3D; testGeometry; testDesktopModel; testPerformance; testVerification;"
```

Run the new MATLAB desktop checks with a graphical display:

```matlab
addpath('matlab', 'matlab/tests'); setupTorch;
testDesktopModel;
testDesktop;
testPerformance;
testVerification;
```

Tests cover SI temperature and cold/hot chemistry, 1D/2D reaction parity, smooth chemical time convergence, positive internal energy and final-time completion across all 1D limiters, periodic species transport, preset overrides, CLI validation, reactive EOS, boundary mass balance and RDE diagnostics. The Python suite checks MATLAB/Python chemistry and reactive-driver parity when Octave is available. A separate smooth entropy-wave test verifies hydrodynamic mesh convergence. Additional regressions cover invalid inputs, integer-coordinate exact references, rejected cross-dimension CLI presets, and shuffled 2D CSV rows. Plot exports require a complete rectangular 2D grid with one row per cell, and preserve dots in the output prefix.

GitHub Actions runs numerical and GUI tests with Python 3.10 and 3.12 and MATLAB R2024a. It also runs the 1D/2D/3D Octave numerical suite and cross-language regression checks. Graphical tests use a virtual display. The v1.4 workflows also run quick analytic studies and preserve their JSON reports as artifacts, alongside informational benchmarks; timing does not gate correctness. The MATLAB workflow additionally exchanges checkpoints with Python in six inert/reactive cases and verifies MATLAB scientific output with independent Python readers. The v1.3 checks exercise both new desktops and resume recorded MATLAB sessions in the Python desktop. The v1.2 workflow adds 27 geometry/boundary continuation cases (all dimensions, inert/Arrhenius and reservoir coupling). Local configuration, credentials, caches and generated outputs are excluded by `.gitignore`.

Independent v1.2 checks cover analytic shell/sector volumes, uniform rest at supported axes/poles, the homologous radial-expansion solution with mesh convergence, rotating equilibrium, an exact Sod reference, characteristic entropy/total-condition invariants, every inlet/outlet face, physical export coordinates and closed-domain mass/energy/species balances. Run `testGeometry` in MATLAB/Octave for its analytic checks.

Checkpoint regressions cover exact continuous/restarted equivalence in every dimension and Python kinetic model, preserved balances and induction variables, schema/data rejection, failed-write atomicity and streamed outputs. MATLAB has its own checkpoint/restart tests; CI also verifies cross-language continuation and VTK/XDMF geometry/fields. Run MATLAB IO tests with `addpath('matlab/tests'); testSession;`.

The 3D tests verify uniform-state preservation, transport of both tangential momenta, Sod tubes along all three axes, axis permutations, reflection symmetry of the explosion for all five limiters, and an extruded 2D case. Further checks cover periodic species and induction transport, all five conservation balances, reflective walls, chemistry parity, invalid inputs and premature completion. Python/Octave comparisons cover inert and reactive 3D cases. The MATLAB suite exercises the same physical invariants with MATLAB R2024a as well as Octave.

For a smooth periodic entropy wave transported along x to `t=0.1`, with four cells in each transverse direction, Van Leer and CFL `0.4`, the 3D solver measured:

| x cells | Density L1 error | Observed L1 order |
| --- | --- | --- |
| 16 | 3.12962e-3 | — |
| 32 | 7.86196e-4 | 1.99 |
| 64 | 1.83072e-4 | 2.10 |

This demonstrates smooth **hydrodynamic** convergence for the tested case. It does not establish second-order accuracy of reactive species transport or validate a quantitative detonation model.

The Sod convergence study reports computed norms and observed orders. A local run with Van Leer, CFL `0.5`, `t=0.2` produced:

| Cells | Density L1 error | Observed L1 order |
| ----- | ---------------- | ----------------- |
| 50    | 2.34816e-2       | —                |
| 100   | 1.53943e-2       | 0.61              |
| 200   | 9.67088e-3       | 0.67              |
| 400   | 6.09489e-3       | 0.67              |

Observed orders depend on the benchmark and resolution.

## 🤝 Acknowledgments

This project was inspired by and developed following the numerical formulation presented by **Paul Augustus Aliaga Brown** in his Bachelor's Thesis [*Desarrollo de herramienta de simulación de una cámara de detonación rotativa*](https://hdl.handle.net/10115/382337), at **Universidad Rey Juan Carlos**. The original research provides the foundation for the MATLAB implementation and the subsequent Python and 2D/3D extensions.

See also [RDE Simulator](https://uk.mathworks.com/matlabcentral/fileexchange/183636-rde-simulator) on MATLAB Central File Exchange.
