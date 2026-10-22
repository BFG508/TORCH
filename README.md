# TORCH 🔥

**T**ransient **O**mni-dimensional **R**eactive **C**omputational **H**ydrodynamics

An experimental Python and MATLAB framework for compressible flow and reduced reactive gas dynamics in **1D, 2D and 3D**. This project numerically solves the Euler equations using finite-volume methods, combining shock-tube benchmarks, reactive-flow simulations and interactive visualization with conservation diagnostics and numerical verification.

## 🚀 Features

* **Finite-Volume Hydrodynamics:** Solves the conservative Euler equations with MUSCL-Hancock reconstruction and HLLC interface fluxes, including tangential momentum transport in 2D and both tangential momenta in 3D.
* **Reactive Gas Dynamics:** Includes Arrhenius chemistry, composition-dependent thermodynamics and conservative reactant transport. Python also supports a reduced induction/exothermic two-stage model through `--kinetic_model znd`.
* **Slope Limiters & Boundary Conditions:** Provides Van Leer, Van Albada, MINBEE, Superbee and bounded Unos limiters, alongside open, reflective and periodic boundaries.
* **Benchmark Simulations:** Includes Sod shock tubes, rarefaction waves, strong shocks, 2D explosions and implosions, quadrant Riemann problems, a reduced annular RDE chamber case, 3D Sod tubes along each axis, spherical explosions and 3D ignition.
* **Interactive Visualization:** Offers Python/Tkinter and MATLAB App Designer interfaces for 1D simulations, with animation controls and flow-property plots. The Python interface also supports an exact Riemann reference overlay.
* **Data Export & Diagnostics:** Exports CSV datasets, JSON metadata and PNG/SVG figures through the Python workflows, with boundary-aware conservation audits, five-equation volume balances in 3D and a 2D exit-thrust diagnostic per unit depth.
* **Numerical Verification:** Includes exact Riemann comparisons, mesh-convergence studies, Python and MATLAB/Octave tests, and cross-language regression checks.

The current version supports **1D, 2D and Cartesian 3D**. The desktop GUIs remain 1D; 3D is available through the Python CLI and Python/MATLAB functions, with plane slices and centerline profiles. The chemistry and RDE boundary model are reduced approximations; quantitative detonation predictions require further physical calibration and validation. Hydrodynamic reconstruction is second order for smooth solutions, while species transport is first order. Equations, units and accuracy limits are documented below.

## 🛠️ Technology Stack

This project is developed in **Python** and **MATLAB**. The Python implementation uses **NumPy** for numerical operations, **Matplotlib** for visualization and **Tkinter** for its desktop interface. **Numba** supplies optional standalone acceleration kernels; the simulation drivers do not automatically use them.

Python **3.10 or newer** is required. The numerical MATLAB functions are also tested with **GNU Octave**, while the `.mlapp` desktop interface requires MATLAB App Designer. Python dependencies are listed in `requirements.txt`; Tkinter may require installation through the operating system.

## 📂 Repository Structure

* `python/` - Python simulation drivers, visualization tools and numerical package.
  * `cfdSolver/torchSolver.py` - Contains the 1D hydrodynamics, thermodynamics, reaction models and simulation presets.
  * `cfdSolver/torchSolver2D.py` - Implements the 2D solver, reactive driver, boundary conditions and RDE diagnostics.
  * `cfdSolver/torchSolver3D.py` - Implements the Cartesian 3D solver, reaction/transport coupling and volume balances.
  * `cfdSolver/torchFlux.py` - Shares directional HLLC fluxes between the 2D and 3D solvers.
  * `cfdSolver/torchPlot3D.py` - Exports 3D volume slices and centerline profiles.
  * `cfdSolver/torchAnalysis.py` - Provides exact Riemann references, error norms, convergence studies and figure exports.
  * **Simulation & Visualization Tools:**
    * `torchCli.py` - Runs batch simulations and exports CSV, JSON and figures.
    * `torchGui.py` - Launches the interactive 1D Python desktop interface.
    * `torchPlot.py` - Exports figures from existing simulation CSV files.
  * `tests/` - Contains numerical, command-line and graphical regression tests.
* `matlab/` - MATLAB implementation and compatibility drivers.
  * `torchSolver.m` / `torchSolver2D.m` / `torchSolver3D.m` - Implement the 1D, 2D and 3D finite-volume solvers.
  * `torchHllcFlux.m` - Provides directional HLLC fluxes for the 3D solver.
  * `plotTorch3D.m` - Displays and exports 3D plane slices and profiles in MATLAB.
  * `torchReaction.m` - Defines the shared SI Arrhenius chemistry model.
  * `inputLoopResult*.m` - Provide inert, reactive, 2D and 3D simulation entry points.
  * `uconsvToUprim*.m` - Convert conservative states into primitive flow properties.
  * `interface1D.mlapp` - Contains the interactive 1D MATLAB App Designer interface.
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
4. **Run a 2D Simulation:** Execute a cylindrical explosion on a custom mesh:

   ```bash
   python3 python/torchCli.py --dim 2d --preset_2d 2d_explosion \
     --nx 40 --ny 40 --t_corte 0.02 \
     --out out/explosion.csv --plot out/explosion
   ```
5. **Run a 3D Simulation:** Execute a spherical explosion and export a central plane slice:

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
   addpath('matlab');
   interface1D;

   % Run a 2D case directly
   [U, t, lambda, X, Y, diagnostic] = inputLoopResult2D( ...
       '2d_explosion', 40, 40, 0.02, 0.4, 'VanLeer');

   % Run and plot a 3D case
   result = torchSolver3D('preset_name','3d_explosion', ...
       'nx',16,'ny',16,'nz',16,'t_end',0.02);
   plotTorch3D(result,'z',[],'out/explosion3d_matlab');
   ```
8. **Run Verification:** Execute the Python and MATLAB/Octave numerical suites:

   ```bash
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
addpath('matlab');

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

MATLAB App Designer can open `matlab/interface1D.mlapp`. Both desktop interfaces have been exercised in a graphical session; repeatable checks are provided in [the MATLAB GUI tests](matlab/tests/testMatlabGui.m). Tk regressions run with the Python suite when a display is accessible and skip otherwise. Run the MATLAB GUI regressions separately:

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

The 3D defaults are 20 cells per axis and CFL `0.4`; the final time defaults to `0.05 s`, or `1e-5 s` for the ignition preset. Each axis supports open, reflective and periodic boundaries. There is no prescribed-inlet or annular-geometry RDE model in this 3D version. The existing 2D RDE and thrust diagnostic retain their scope.

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

## ✅ Verification

```bash
python3 -m unittest discover -s python/tests -v
octave --no-gui --quiet --eval "run('matlab/tests/testMatlabSolver.m'); addpath('matlab/tests'); testMatlabSolver3D;"
```

Tests cover SI temperature and cold/hot chemistry, 1D/2D reaction parity, smooth chemical time convergence, positive internal energy and final-time completion across all 1D limiters, periodic species transport, preset overrides, CLI validation, reactive EOS, boundary mass balance and RDE diagnostics. The Python suite checks MATLAB/Python chemistry and reactive-driver parity when Octave is available. A separate smooth entropy-wave test verifies hydrodynamic mesh convergence. Additional regressions cover invalid inputs, integer-coordinate exact references, rejected cross-dimension CLI presets, and shuffled 2D CSV rows. Plot exports require a complete rectangular 2D grid with one row per cell, and preserve dots in the output prefix.

GitHub Actions runs numerical and GUI tests with Python 3.10 and 3.12 and MATLAB R2024a. It also runs the 1D/2D/3D Octave numerical suite and cross-language regression checks. Graphical tests use a virtual display. Local configuration, credentials, caches and generated outputs are excluded by `.gitignore`.

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
