% Automated test suite for the MATLAB TORCH CFD Solver
% Runs non-reactive and reactive simulations and verifies numerical validity.

script_path = which(mfilename);
if isempty(script_path)
    script_path = mfilename('fullpath');
end
if ~isempty(script_path) && ~strcmp(script_path, mfilename)
    script_dir = fileparts(script_path);
    if exist(script_dir, 'dir')
        addpath(script_dir);
        parent_dir = fileparts(script_dir);
        if exist(parent_dir, 'dir')
            addpath(parent_dir);
        end
    end
end

fprintf('========================================\n');
fprintf(' Running MATLAB / Octave CFD Test Suite \n');
fprintf('========================================\n\n');

num_passed = 0;
num_total  = 0;

limiters = {'VanLeer', 'Albada', 'MINBEE', 'Superbee', 'Unos'};
wave_speeds = {'Pressure Based', 'Enthalpy Based'};

% Test 1-10: Non-reactive Sod Shock Tube across limiters and wave speed formulations
for k_lim = 1:numel(limiters)
    for k_ws = 1:numel(wave_speeds)
        lim = limiters{k_lim};
        ws  = wave_speeds{k_ws};
        num_total = num_total + 1;
        fprintf('[Test %02d] Non-reactive Sod | Limiter: %-7s | WS: %-14s ... ', num_total, lim, ws);

        [U_evo, t] = inputLoopResult(1.0, 50, ...
            1.0, 0.0, 1.0, 0.125, 0.0, 0.1, ...
            1.4, 0.5, 0.05, ...
            lim, 'Sound Speed Controlled', ...
            'Variante 1 - Original', ws, 'Open', 50);

        assert(~any(isnan(U_evo(:))), 'NaN detected in U_evo');
        assert(~any(isinf(U_evo(:))), 'Inf detected in U_evo');
        density = U_evo(1, :, :);
        assert(all(density(:) > 0), 'Non-positive density detected');
        assert(numel(t) > 1, 'Simulation did not advance in time');
        assert(abs(t(end)-0.05)<1e-12, 'Simulation ended prematurely');
        internal = squeeze(U_evo(3,:,:)-0.5.*U_evo(2,:,:).^2./U_evo(1,:,:));
        assert(all(internal(:)>0), 'Nonpositive internal energy');

        fprintf('PASSED (%d steps, t=%.4f)\n', numel(t)-1, t(end));
        num_passed = num_passed + 1;
    end
end

% Test 11: Reflective boundary conditions
num_total = num_total + 1;
fprintf('[Test %02d] Non-reactive Sod | Reflective BC               ... ', num_total);
[U_evo, t] = inputLoopResult(1.0, 50, ...
    1.0, 0.0, 1.0, 0.125, 0.0, 0.1, ...
    1.4, 0.5, 0.05, ...
    'VanLeer', 'Sound Speed Controlled', ...
    'Variante 1 - Original', 'Pressure Based', 'Reflective', 50);

assert(~any(isnan(U_evo(:))), 'NaN detected in Reflective BC');
assert(~any(isinf(U_evo(:))), 'Inf detected in Reflective BC');
fprintf('PASSED (%d steps)\n', numel(t)-1);
num_passed = num_passed + 1;

% Test 12-21: Reactive Detonation across limiters and wave speed formulations
for k_lim = 1:numel(limiters)
    for k_ws = 1:numel(wave_speeds)
        lim = limiters{k_lim};
        ws  = wave_speeds{k_ws};
        num_total = num_total + 1;
        fprintf('[Test %02d] Reactive H2/O2   | Limiter: %-7s | WS: %-14s ... ', num_total, lim, ws);

        lambda_init = ones(1, 50);
        [U_evo, t, lambda_evo] = inputLoopResultwQR(1.0, 50, ...
            1.0, 0.0, 1e6, 0.5, 0.0, 101325.0, ...
            lambda_init, 0.5, 1e-5, ...
            lim, 'Sound Speed Controlled', ...
            'Variante 1 - Original', ws, 'Open', 50);

        assert(~any(isnan(U_evo(:))), 'NaN detected in reactive U_evo');
        assert(~any(isinf(U_evo(:))), 'Inf detected in reactive U_evo');
        assert(~any(isnan(lambda_evo(:))), 'NaN detected in lambda_evo');
        assert(all(lambda_evo(:) >= 0.0) && all(lambda_evo(:) <= 1.0), 'lambda out of bounds [0, 1]');
        assert(numel(t) > 1, 'Reactive simulation did not advance');
        assert(abs(t(end)-1e-5)<1e-12, 'Reactive simulation ended prematurely');
        assert(min(lambda_evo(end,:))<0.99,'Hot mixture did not react');

        fprintf('PASSED (%d steps, t=%.5f)\n', numel(t)-1, t(end));
        num_passed = num_passed + 1;
    end
end

% =========================================================================
% 2D SUITE: HYDRODYNAMICS & KINETICS
% =========================================================================

% [Test 22] 2D Primitive-Conservative Conversion
num_total = num_total + 1;
fprintf('[Test %02d] 2D Primitive/Conservative Conversion ... ', num_total);
rho_orig = 1.2 .* ones(8, 8);
u_orig   = 40.0 .* ones(8, 8);
v_orig   = -20.0 .* ones(8, 8);
p_orig   = 101325.0 .* ones(8, 8);
Uprim_orig = zeros(4, 8, 8);
Uprim_orig(1, :, :) = rho_orig;
Uprim_orig(2, :, :) = u_orig;
Uprim_orig(3, :, :) = v_orig;
Uprim_orig(4, :, :) = p_orig;

Uconsv_test = zeros(4, 8, 8);
Uconsv_test(1, :, :) = rho_orig;
Uconsv_test(2, :, :) = rho_orig .* u_orig;
Uconsv_test(3, :, :) = rho_orig .* v_orig;
Uconsv_test(4, :, :) = p_orig ./ (1.4 - 1.0) + 0.5 .* rho_orig .* (u_orig.^2 + v_orig.^2);

Uprim_rec = uconsvToUprim2D(Uconsv_test, 1.4);
assert(max(abs(Uprim_rec(:) - Uprim_orig(:))) < 1e-10, '2D conversion mismatch');
fprintf('PASSED\n');
num_passed = num_passed + 1;

% [Test 23] 2D Cylindrical Explosion (VanLeer)
num_total = num_total + 1;
fprintf('[Test %02d] 2D Explosion (Limiter: VanLeer) ... ', num_total);
[U_exp, t_exp] = torchSolver2D('preset', '2d_explosion', 16, 16, 0.02, 0.4, 'VanLeer');
assert(~any(isnan(U_exp(:))), 'NaN in 2D explosion');
assert(~any(isinf(U_exp(:))), 'Inf in 2D explosion');
assert(all(U_exp(1, :) > 0), 'Negative density in 2D explosion');
fprintf('PASSED (%d steps, t=%.4f)\n', numel(t_exp)-1, t_exp(end));
num_passed = num_passed + 1;

% [Test 24] 2D Cylindrical Explosion (Superbee)
num_total = num_total + 1;
fprintf('[Test %02d] 2D Explosion (Limiter: Superbee) ... ', num_total);
[U_exp2, t_exp2] = torchSolver2D('preset', '2d_explosion', 16, 16, 0.02, 0.4, 'Superbee');
assert(~any(isnan(U_exp2(:))), 'NaN in 2D Superbee explosion');
fprintf('PASSED (%d steps, t=%.4f)\n', numel(t_exp2)-1, t_exp2(end));
num_passed = num_passed + 1;

% [Test 25] 2D Reactive RDE Chamber (Periodic BC, Strang Splitting)
num_total = num_total + 1;
fprintf('[Test %02d] 2D Reactive RDE Chamber (Periodic BC) ... ', num_total);
[U_rde, t_rde, lam_rde] = inputLoopResult2D('2d_rde_chamber', 16, 16, 0.001, 0.35, 'VanLeer');
assert(~any(isnan(U_rde(:))), 'NaN in 2D RDE chamber');
assert(abs(t_rde(end)-0.001)<1e-12, '2D simulation ended prematurely');
assert(~any(isnan(lam_rde(:))), 'NaN in 2D RDE lambda');
assert(all(lam_rde(:) >= 0.0) && all(lam_rde(:) <= 1.0), '2D RDE lambda out of bounds');
fprintf('PASSED (%d steps, t=%.5f)\n', numel(t_rde)-1, t_rde(end));
num_passed = num_passed + 1;

% Regression: chemistry must not react rapidly at 300 K.
num_total = num_total + 1;
rho = 101325*12.0106666666667/(8314.46*300);
U = [rho;0;101325/0.4];
[lam, reacted] = torchReaction(U,1,1e-6);
assert(lam>1-1e-10,'Cold chemistry has incorrect SI units');
assert(abs(reacted(3)-U(3)-141584000*rho*(1-lam))<1e-8,'Chemical energy balance mismatch');
num_passed = num_passed + 1;

% Regression: periodic species must move with the fluid and conserve rho*lambda.
num_total = num_total + 1;
lam0 = zeros(1,60); lam0(11:20)=1;
[U,t,lam] = torchSolver('reactive',1,60,1,2,100,1,2,100,lam0,0.5,0.03,'VanLeer','Sound Speed Controlled','Variante 1 - Original','Pressure Based','Periodic',50);
assert(abs(sum(U(1,:,end).*lam(end,:))-sum(lam0))<1e-9,'Species transport is not conservative');
assert(sum((1:60).*U(1,:,end).*lam(end,:))>sum((1:60).*lam0),'Species did not advect');
num_passed = num_passed + 1;

% Regression: fully burned composition uses the same gamma as Python.
num_total = num_total + 1;
[U,~,~] = torchSolver('reactive',1,8,1,0,100,1,0,100,zeros(1,8),0.5,0,'VanLeer','Sound Speed Controlled','Variante 1 - Original','Pressure Based','Periodic',50);
assert(max(abs(U(3,:)-100/(9/7-1)))<1e-10,'Burned gas gamma differs across engines');
num_passed = num_passed + 1;

% Regression: open-domain conservation includes the inlet/outlet mass flux.
num_total = num_total + 1;
[~,t,~,~,~,diagnostic] = torchSolver2D('preset','2d_rde_chamber',10,10,2e-6,0.4,'VanLeer');
assert(abs(t(end)-2e-6)<1e-12,'RDE ended prematurely');
assert(diagnostic.mass_conservation_error<1e-12,'Boundary mass balance mismatch');
assert(isfinite(diagnostic.exit_thrust_per_depth),'Invalid exit thrust');
num_passed = num_passed + 1;

% Regression: reject invalid primitive states and chemistry before a zero-time return.
base = {'inert',1,8,1,0,1,0.125,0,0.1,1.4,0.4,0,'VanLeer','Sound Speed Controlled','Variante 1 - Original','Pressure Based','Open',50};
for bad_index = [2,3,4,6,10]
    args = base;
    args{bad_index} = -1;
    rejected = false;
    try
        torchSolver(args{:});
    catch
        rejected = true;
    end
    assert(rejected, 'Invalid initial input accepted');
    num_total = num_total + 1;
    num_passed = num_passed + 1;
end
args = base; args{1} = 'reactive'; args{10} = [ones(1,7),NaN];
rejected = false;
try
    torchSolver(args{:});
catch
    rejected = true;
end
assert(rejected, 'Nonfinite reactant fraction accepted');
num_total = num_total + 1; num_passed = num_passed + 1;

fprintf('\n========================================\n');
fprintf(' ALL %d MATLAB/OCTAVE TESTS PASSED!\n', num_passed);
fprintf('========================================\n');
