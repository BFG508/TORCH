function [U_final, t_arr, lambda_final, X, Y, diagnostics] = torchSolver2D(varargin)
% TORCHSOLVER2D Unified 2D Finite-Volume CFD Solver for Compressible Reactive Gas Dynamics.
%
% Solves the 2D unsteady Euler equations with unsplit 2nd-order MUSCL-Hancock
% spatial reconstruction, Sweby-bounded TVD slope limiters, directional HLLC
% approximate Riemann solver, and 2nd-order Strang operator splitting for
% finite-rate H2/O2 chemical kinetics.
%
% Calling syntaxes:
%   1) Using preset:
%      [U_final, t_arr, lambda_final, X, Y] = torchSolver2D('preset', preset_name, nx, ny, t_end, cfl, limiter)
%   2) Custom setup:
%      [U_final, t_arr, lambda_final, X, Y] = torchSolver2D('custom', U_init, lx, ly, nx, ny, t_end, cfl, limiter, bc_x, bc_y, is_reactive, lambda_init)
%
% Author: BFG508 (MATLAB implementation; original 1D provenance by Paul Augustus Aliaga Brown)

if nargin>0 && any(strcmp(varargin{1},{'session_dt','session_step'}))
    U=varargin{2}; gamma=varargin{3}; h=varargin{4}; c=varargin{5};
    if strcmp(varargin{1},'session_dt')
        U_final=paddedCfl(U,gamma,c.boundaries_xyz{1},c.boundaries_xyz{2},c.inlet_primitive,h(1),h(2),c.cfl);
        t_arr=[];
    else
        [U_final,Fx,Fy]=musclHancockStep2D(U,h(1),h(2),varargin{6},gamma,c.limiter,c.boundaries_xyz{1},c.boundaries_xyz{2},c.inlet_primitive);
        t_arr={Fx,Fy};
    end
    lambda_final=[]; X=[]; Y=[]; diagnostics=[]; return
end

if nargin == 0
    error('torchSolver2D requires simulation parameters.');
end

mode = varargin{1};
inlet_primitive = [1,0,100,101325];
ambient_pressure = 101325;

if strcmpi(mode, 'preset')
    preset_name = varargin{2};
    nx = 40; ny = 40; t_end = 0.2; cfl = 0.4; limiter = 'VanLeer';
    if nargin >= 3, nx = varargin{3}; end
    if nargin >= 4, ny = varargin{4}; end
    if nargin >= 5, t_end = varargin{5}; end
    if nargin >= 6, cfl = varargin{6}; end
    if nargin >= 7, limiter = varargin{7}; end
    if nargin >= 8, inlet_primitive = varargin{8}; end
    if nargin >= 9, ambient_pressure = varargin{9}; end

    [U_init, lx, ly, bc_x, bc_y, is_reactive, lambda_init] = get2DPreset(preset_name, nx, ny);

elseif strcmpi(mode, 'custom')
    U_init      = varargin{2};
    lx          = varargin{3};
    ly          = varargin{4};
    nx          = varargin{5};
    ny          = varargin{6};
    t_end       = varargin{7};
    cfl         = varargin{8};
    limiter     = varargin{9};
    bc_x        = varargin{10};
    bc_y        = varargin{11};
    is_reactive = false;
    lambda_init = [];
    if nargin >= 12, is_reactive = varargin{12}; end
    if nargin >= 13, lambda_init = varargin{13}; end
    if nargin >= 14, inlet_primitive = varargin{14}; end
    if nargin >= 15, ambient_pressure = varargin{15}; end
else
    error('Unknown mode: %s. Use "preset" or "custom".', mode);
end

assert(all(isfinite([lx,ly,nx,ny,cfl,t_end])) && lx>0 && ly>0 && nx>=2 && ny>=2 && nx==fix(nx) && ny==fix(ny) && cfl>0 && cfl<=0.5 && t_end>=0, 'Invalid domain/grid/CFL/time');
assert(isequal(size(U_init),[4,ny,nx]) && physicalState2D(U_init), 'Invalid initial conservative state');
assert(isscalar(ambient_pressure) && isfinite(ambient_pressure) && ambient_pressure>=0, 'Invalid ambient pressure');
if is_reactive
    assert(isequal(size(lambda_init),[ny,nx]) && all(isfinite(lambda_init(:))) && all(lambda_init(:)>=0 & lambda_init(:)<=1), 'Invalid initial reactant fraction');
end
dx = lx / nx;
dy = ly / ny;
x_coords = linspace(0.5 * dx, lx - 0.5 * dx, nx);
y_coords = linspace(0.5 * dy, ly - 0.5 * dy, ny);
[X, Y] = meshgrid(x_coords, y_coords);

gamma = 1.4;
U_curr = U_init;
lambda_curr = lambda_init;

current_time = 0.0;
step_count = 0;
max_steps = 100000;
t_arr = current_time;
boundary_mass = 0;
initial_mass = sum(reshape(U_curr(1,:,:),[],1))*dx*dy;

while current_time < t_end && step_count < max_steps
    if is_reactive, gamma = (4.5-lambda_curr)./(3.5-lambda_curr); end
    dt = paddedCfl(U_curr,gamma,bc_x,bc_y,inlet_primitive,dx,dy,cfl);

    if current_time + dt > t_end
        dt = t_end - current_time;
    end

    if ~isfinite(dt) || dt <= 1e-15
        error('Time integration stalled before requested final time');
    end

    if is_reactive
        cfl_stable = false;
        for retry = 1:60
            [heated, lam_half] = reactionStep2D(U_curr, lambda_curr, 0.5*dt);
            gp = (4.5-lam_half)./(3.5-lam_half);
            allowed = paddedCfl(heated,gp,bc_x,bc_y,inlet_primitive,dx,dy,cfl);
            if dt <= allowed*(1+1e-12), cfl_stable = true; break; end
            dt = min(0.5*dt,allowed);
        end
        if ~cfl_stable, error('Could not find a CFL-stable reactive step'); end
        U_curr = heated; lambda_curr = lam_half;
    end

    % Convective Hydrodynamic Step: Full step dt via 2D MUSCL-Hancock
    if is_reactive, gamma = (4.5-lambda_curr)./(3.5-lambda_curr); end
    U_old = U_curr;
    [U_curr, flux_x, flux_y] = musclHancockStep2D(U_curr, dx, dy, dt, gamma, limiter, bc_x, bc_y, inlet_primitive);
    boundary_mass = boundary_mass + dt*(dy*sum(reshape(flux_x(1,:,end)-flux_x(1,:,1),[],1)) + dx*sum(reshape(flux_y(1,end,:)-flux_y(1,1,:),[],1)));
    if is_reactive
        lambda_curr = advectScalar2D(lambda_curr, U_old, U_curr, flux_x, flux_y, dt/dx, dt/dy, bc_x, bc_y);
    end

    if is_reactive
        % Strang Splitting Step 2: Chemical half-step dt/2
        [U_curr, lambda_curr] = reactionStep2D(U_curr, lambda_curr, 0.5 * dt);
    end

    current_time = current_time + dt;
    step_count = step_count + 1;
    t_arr(end + 1) = current_time; %#ok<AGROW>
end

if current_time < t_end, error('Maximum steps reached before requested final time'); end
U_final = U_curr;
lambda_final = lambda_curr;
if is_reactive, gamma = (4.5-lambda_curr)./(3.5-lambda_curr); end
prim_final = uconsvToUprim2D(U_final,gamma);
rho_exit = reshape(prim_final(1,end,:),1,[]);
v_exit = reshape(prim_final(3,end,:),1,[]);
p_exit = reshape(prim_final(4,end,:),1,[]);
diagnostics.exit_thrust_per_depth = dx*sum(rho_exit.*v_exit.^2+p_exit-ambient_pressure);
final_mass = sum(reshape(U_final(1,:,:),[],1))*dx*dy;
diagnostics.mass_conservation_error = abs(final_mass-initial_mass+boundary_mass)/initial_mass;
diagnostics.net_boundary_mass = boundary_mass;

end


% =========================================================================
% PRESETS GENERATOR (2D)
% =========================================================================

function [U_init, lx, ly, bc_x, bc_y, is_reactive, lambda_init] = get2DPreset(pname, nx, ny)
lx = 1.0; ly = 1.0;
dx = lx / nx; dy = ly / ny;
x = linspace(0.5 * dx, lx - 0.5 * dx, nx);
y = linspace(0.5 * dy, ly - 0.5 * dy, ny);
[X, Y] = meshgrid(x, y);

gamma = 1.4;
is_reactive = false;
lambda_init = [];

switch lower(pname)
    case {'2d_explosion', 'explosion', 'cylindrical_explosion'}
        radius = sqrt((X - 0.5).^2 + (Y - 0.5).^2);
        rho = ones(ny, nx) .* 0.125;
        rho(radius <= 0.2) = 1.0;
        p = ones(ny, nx) .* 0.1;
        p(radius <= 0.2) = 1.0;
        u = zeros(ny, nx);
        v = zeros(ny, nx);
        bc_x = 'Transmissive';
        bc_y = 'Transmissive';

    case {'2d_implosion', 'implosion'}
        radius = sqrt((X - 0.5).^2 + (Y - 0.5).^2);
        rho = ones(ny, nx) .* 1.0;
        rho(radius <= 0.2) = 0.125;
        p = ones(ny, nx) .* 1.0;
        p(radius <= 0.2) = 0.14;
        u = zeros(ny, nx);
        v = zeros(ny, nx);
        bc_x = 'Reflective';
        bc_y = 'Reflective';

    case {'2d_riemann_quadrant', 'riemann_4quadrant'}
        q1 = (X >= 0.5) & (Y >= 0.5);
        q2 = (X < 0.5) & (Y >= 0.5);
        q3 = (X < 0.5) & (Y < 0.5);
        q4 = (X >= 0.5) & (Y < 0.5);
        rho = zeros(ny, nx); u = zeros(ny, nx); v = zeros(ny, nx); p = zeros(ny, nx);
        rho(q1) = 0.5313; p(q1) = 0.4; u(q1) = 0.0; v(q1) = 0.0;
        rho(q2) = 1.0;    p(q2) = 1.0; u(q2) = 0.7276; v(q2) = 0.0;
        rho(q3) = 0.8;    p(q3) = 1.0; u(q3) = 0.0; v(q3) = 0.0;
        rho(q4) = 1.0;    p(q4) = 1.0; u(q4) = 0.0; v(q4) = 0.7276;
        bc_x = 'Transmissive';
        bc_y = 'Transmissive';

    case {'2d_shock_tube', 'planar_shock'}
        rho = ones(ny, nx) .* 0.125;
        rho(X <= 0.5) = 1.0;
        p = ones(ny, nx) .* 0.1;
        p(X <= 0.5) = 1.0;
        u = zeros(ny, nx);
        v = zeros(ny, nx);
        bc_x = 'Transmissive';
        bc_y = 'Reflective';

    case {'2d_rde_chamber', 'rde_detonation'}
        dist = sqrt((X - 0.2).^2 + (Y - 0.1).^2);
        rho = ones(ny, nx) .* 1.0;
        rho(dist <= 0.1) = 2.5;
        u = zeros(ny, nx);
        u(dist <= 0.1) = 500.0;
        v = zeros(ny, nx);
        p = ones(ny, nx) .* 1.0;
        p(dist <= 0.1) = 30.0;
        p = p .* 101325;
        bc_x = 'Periodic';
        bc_y = 'Inflow';
        is_reactive = true;
        lambda_init = ones(ny, nx);

    otherwise
        error('Unknown preset: %s', pname);
end

U_init = zeros(4, ny, nx);
U_init(1, :, :) = rho;
U_init(2, :, :) = rho .* u;
U_init(3, :, :) = rho .* v;
kinetic = 0.5 .* rho .* (u.^2 + v.^2);
internal = p ./ (gamma - 1.0);
U_init(4, :, :) = internal + kinetic;

end


% =========================================================================
% BOUNDARY CONDITIONS (2D)
% =========================================================================

function Upad = applyBc2D(U, bc_x, bc_y, inlet_primitive)
if nargin < 4, inlet_primitive = [1,0,100,101325]; end
[nvars, ny, nx] = size(U);
Upad = zeros(nvars, ny + 4, nx + 4);
Upad(:, 3:ny+2, 3:nx+2) = U;

% X boundaries
switch lower(bc_x)
    case {'periodic', 'wrap'}
        Upad(:, 3:ny+2, 1) = U(:, :, end-1);
        Upad(:, 3:ny+2, 2) = U(:, :, end);
        Upad(:, 3:ny+2, end-1) = U(:, :, 1);
        Upad(:, 3:ny+2, end)   = U(:, :, 2);
    case {'open', 'transmissive', 'outflow'}
        Upad(:, 3:ny+2, 1) = U(:, :, 1);
        Upad(:, 3:ny+2, 2) = U(:, :, 1);
        Upad(:, 3:ny+2, end-1) = U(:, :, end);
        Upad(:, 3:ny+2, end)   = U(:, :, end);
    case {'reflective', 'wall', 'slip'}
        Upad(:, 3:ny+2, 2) = U(:, :, 1);
        Upad(2, 3:ny+2, 2) = -U(2, :, 1); % Invert normal u
        Upad(:, 3:ny+2, 1) = U(:, :, 2);
        Upad(2, 3:ny+2, 1) = -U(2, :, 2);

        Upad(:, 3:ny+2, end-1) = U(:, :, end);
        Upad(2, 3:ny+2, end-1) = -U(2, :, end);
        Upad(:, 3:ny+2, end)   = U(:, :, end-1);
        Upad(2, 3:ny+2, end)   = -U(2, :, end-1);
end

% Y boundaries
switch lower(bc_y)
    case {'periodic', 'wrap'}
        Upad(:, 1, :) = Upad(:, ny+1, :);
        Upad(:, 2, :) = Upad(:, ny+2, :);
        Upad(:, end-1, :) = Upad(:, 3, :);
        Upad(:, end, :)   = Upad(:, 4, :);
    case 'inflow'
        q = inlet_primitive;
        assert(numel(q)==4 && all(isfinite(q)) && q(1)>0 && q(4)>0 && q(3)>=0,'Invalid inlet primitive state');
        inlet = [q(1); q(1)*q(2); q(1)*q(3); q(4)/(1.4-1)+0.5*q(1)*(q(2)^2+q(3)^2)];
        Upad(:,1:2,:) = repmat(reshape(inlet,[4,1,1]),[1,2,nx+4]);
        Upad(:,end-1,:) = Upad(:,ny+2,:);
        Upad(:,end,:) = Upad(:,ny+2,:);
    case {'open', 'transmissive', 'outflow'}
        Upad(:, 1, :) = Upad(:, 3, :);
        Upad(:, 2, :) = Upad(:, 3, :);
        Upad(:, end-1, :) = Upad(:, ny+2, :);
        Upad(:, end, :)   = Upad(:, ny+2, :);
    case {'reflective', 'wall', 'slip'}
        Upad(:, 2, :) = Upad(:, 3, :);
        Upad(3, 2, :) = -Upad(3, 3, :); % Invert normal v
        Upad(:, 1, :) = Upad(:, 4, :);
        Upad(3, 1, :) = -Upad(3, 4, :);

        Upad(:, end-1, :) = Upad(:, ny+2, :);
        Upad(3, end-1, :) = -Upad(3, ny+2, :);
        Upad(:, end, :)   = Upad(:, ny+1, :);
        Upad(3, end, :)   = -Upad(3, ny+1, :);
end
end


% =========================================================================
% TVD SLOPES (2D)
% =========================================================================

function delta = computeSlopes2D(arr, dim, limiter)
% Computes limited slopes along dim (2 for x, 1 for y)
if dim == 2
    % along x (axis 3 in 3D array)
    d_minus = arr(:, :, 2:end-1) - arr(:, :, 1:end-2);
    d_plus  = arr(:, :, 3:end)   - arr(:, :, 2:end-1);
else
    % along y (axis 2 in 3D array)
    d_minus = arr(:, 2:end-1, :) - arr(:, 1:end-2, :);
    d_plus  = arr(:, 3:end, :)   - arr(:, 2:end-1, :);
end

safe_d_plus = d_plus;
mask_zero = abs(d_plus) < 1e-14;
safe_d_plus(mask_zero) = 1e-14;

r = d_minus ./ safe_d_plus;
r_pos = max(r, 0.0);

switch lower(limiter)
    case {'vanleer', 'van_leer'}
        phi = zeros(size(r));
        mask = r > 0.0;
        phi(mask) = (2.0 .* r_pos(mask)) ./ (1.0 + r_pos(mask));
    case {'albada', 'vanalbada', 'van_albada'}
        phi = zeros(size(r));
        mask = r > 0.0;
        phi(mask) = (r_pos(mask).^2 + r_pos(mask)) ./ (r_pos(mask).^2 + 1.0);
    case {'minbee', 'minmod'}
        phi = max(0.0, min(r, 1.0));
    case 'superbee'
        phi = max(0.0, max(min(2.0 .* r, 1.0), min(r, 2.0)));
    case 'unos'
        phi = max(0.0, min(min(2.0, 2.0 .* r), (2.0 + r) ./ 3.0));
    otherwise
        phi = zeros(size(r));
        mask = r > 0.0;
        phi(mask) = (2.0 .* r_pos(mask)) ./ (1.0 + r_pos(mask));
end

delta = phi .* d_plus;
end


% =========================================================================
% DIRECTIONAL HLLC RIEMANN FLUX (1D INTERFACE SLICE)
% =========================================================================

function flux = hllcFluxSlice(u_l, u_r, gamma_l, gamma_r)
if nargin < 4, gamma_r = gamma_l; end
gamma = 0.5 .* (gamma_l + gamma_r);
% u_l, u_r: [4 x M]
rho_l = max(u_l(1, :), 1e-12);
rho_r = max(u_r(1, :), 1e-12);

un_l = u_l(2, :) ./ rho_l;
un_r = u_r(2, :) ./ rho_r;

ut_l = u_l(3, :) ./ rho_l;
ut_r = u_r(3, :) ./ rho_r;

ke_l = 0.5 .* rho_l .* (un_l.^2 + ut_l.^2);
ke_r = 0.5 .* rho_r .* (un_r.^2 + ut_r.^2);

p_l = max((gamma_l - 1.0) .* (u_l(4, :) - ke_l), 1e-12);
p_r = max((gamma_r - 1.0) .* (u_r(4, :) - ke_r), 1e-12);

a_l = sqrt(gamma_l .* p_l ./ rho_l);
a_r = sqrt(gamma_r .* p_r ./ rho_r);

% Physical fluxes
f_l = zeros(size(u_l));
f_l(1, :) = rho_l .* un_l;
f_l(2, :) = rho_l .* un_l.^2 + p_l;
f_l(3, :) = rho_l .* un_l .* ut_l;
f_l(4, :) = un_l .* (u_l(4, :) + p_l);

f_r = zeros(size(u_r));
f_r(1, :) = rho_r .* un_r;
f_r(2, :) = rho_r .* un_r.^2 + p_r;
f_r(3, :) = rho_r .* un_r .* ut_r;
f_r(4, :) = un_r .* (u_r(4, :) + p_r);

% Roe averages
sqrt_l = sqrt(rho_l);
sqrt_r = sqrt(rho_r);
inv_d = 1.0 ./ (sqrt_l + sqrt_r);

u_roe = (sqrt_l .* un_l + sqrt_r .* un_r) .* inv_d;
v_roe = (sqrt_l .* ut_l + sqrt_r .* ut_r) .* inv_d;
h_l   = (u_l(4, :) + p_l) ./ rho_l;
h_r   = (u_r(4, :) + p_r) ./ rho_r;
h_roe = (sqrt_l .* h_l + sqrt_r .* h_r) .* inv_d;
ke_roe = 0.5 .* (u_roe.^2 + v_roe.^2);
a_roe = sqrt(max((gamma - 1.0) .* (h_roe - ke_roe), 1e-12));

% Roe-Einfeldt bounds
s_l = min(un_l - a_l, u_roe - a_roe);
s_r = max(un_r + a_r, u_roe + a_roe);

% Middle wave speed S_*
denom_star = rho_l .* (s_l - un_l) - rho_r .* (s_r - un_r);
denom_safe = denom_star;
denom_safe(abs(denom_star) < 1e-14) = 1e-14;

s_star = (p_r - p_l + rho_l .* un_l .* (s_l - un_l) - rho_r .* un_r .* (s_r - un_r)) ./ denom_safe;

% Star states
denom_sl_ss = s_l - s_star; denom_sl_ss(abs(denom_sl_ss) < 1e-14) = 1e-14;
denom_sr_ss = s_r - s_star; denom_sr_ss(abs(denom_sr_ss) < 1e-14) = 1e-14;

rho_sl = max(rho_l .* ((s_l - un_l) ./ denom_sl_ss), 1e-12);
rho_sr = max(rho_r .* ((s_r - un_r) ./ denom_sr_ss), 1e-12);

denom_sl_un = s_l - un_l; denom_sl_un(abs(denom_sl_un) < 1e-14) = 1e-14;
denom_sr_un = s_r - un_r; denom_sr_un(abs(denom_sr_un) < 1e-14) = 1e-14;

u_star_l = zeros(size(u_l));
u_star_l(1, :) = rho_sl;
u_star_l(2, :) = rho_sl .* s_star;
u_star_l(3, :) = rho_sl .* ut_l;
u_star_l(4, :) = rho_sl .* (u_l(4, :) ./ rho_l + (s_star - un_l) .* (s_star + p_l ./ (rho_l .* denom_sl_un)));

u_star_r = zeros(size(u_r));
u_star_r(1, :) = rho_sr;
u_star_r(2, :) = rho_sr .* s_star;
u_star_r(3, :) = rho_sr .* ut_r;
u_star_r(4, :) = rho_sr .* (u_r(4, :) ./ rho_r + (s_star - un_r) .* (s_star + p_r ./ (rho_r .* denom_sr_un)));

f_star_l = f_l + bsxfun(@times, s_l, (u_star_l - u_l));
f_star_r = f_r + bsxfun(@times, s_r, (u_star_r - u_r));

% Star bypass
mask_close_l = abs(s_l - s_star) < 1e-12;
mask_close_r = abs(s_r - s_star) < 1e-12;
f_star_l(:, mask_close_l) = f_l(:, mask_close_l);
f_star_r(:, mask_close_r) = f_r(:, mask_close_r);

flux = zeros(size(u_l));
mask_l = 0.0 <= s_l;
mask_star_l = (s_l <= 0.0) & (0.0 <= s_star);
mask_star_r = (s_star <= 0.0) & (0.0 <= s_r);
mask_r = 0.0 >= s_r;

flux(:, mask_l) = f_l(:, mask_l);
flux(:, mask_star_l) = f_star_l(:, mask_star_l);
flux(:, mask_star_r) = f_star_r(:, mask_star_r);
flux(:, mask_r) = f_r(:, mask_r);
end


% =========================================================================
% UNSPLIT 2D MUSCL-HANCOCK TIME STEP
% =========================================================================

function [Unext, flux_x, flux_y] = musclHancockStep2D(Uinner, dx, dy, dt, gamma, limiter, bc_x, bc_y, inlet_primitive)
[~, ny, nx] = size(Uinner);

% 1. Ghost cells
Upad = applyBc2D(Uinner, bc_x, bc_y, inlet_primitive);

% Active region
Uactive = Upad(:, 2:ny+3, 2:nx+3);

g_inner = gamma .* ones(ny,nx);
gstate = repmat(reshape(g_inner, [1,ny,nx]), [4,1,1]);
scalar_y = bc_y;
if strcmpi(bc_y,'Inflow'), scalar_y = 'Open'; end
gpad = applyBc2D(gstate, bc_x, scalar_y);
if strcmpi(bc_y,'Inflow'), gpad(1,1:2,:) = 1.4; end
g_active = squeeze(gpad(1,2:ny+3,2:nx+3));
gamma = g_active;

% 2. Limited slopes
delta_x = computeSlopes2D(Upad(:, 2:ny+3, :), 2, limiter);
delta_y = computeSlopes2D(Upad(:, :, 2:nx+3), 1, limiter);

% Predictor boundary extrapolations
u_xl = Uactive - 0.5 .* delta_x;
u_xr = Uactive + 0.5 .* delta_x;
u_yb = Uactive - 0.5 .* delta_y;
u_yt = Uactive + 0.5 .* delta_y;

% Physical Euler fluxes
fx_r = eulerFluxX(u_xr, gamma);
fx_l = eulerFluxX(u_xl, gamma);
fy_t = eulerFluxY(u_yt, gamma);
fy_b = eulerFluxY(u_yb, gamma);

% Predictor update to t + dt/2
Ubar = Uactive - (0.5 * dt / dx) .* (fx_r - fx_l) - (0.5 * dt / dy) .* (fy_t - fy_b);

% Positivity floors
Ubar(1, :, :) = max(Ubar(1, :, :), 1e-12);
r_rho = squeeze(Ubar(1, :, :));
r_u = squeeze(Ubar(2, :, :)) ./ r_rho;
r_v = squeeze(Ubar(3, :, :)) ./ r_rho;
p_bar = (gamma - 1.0) .* (squeeze(Ubar(4, :, :)) - 0.5 .* r_rho .* (r_u.^2 + r_v.^2));
mask_p = p_bar < 1e-12;
if any(mask_p(:))
    E_corr = 1e-12 ./ (gamma - 1.0) + 0.5 .* r_rho .* (r_u.^2 + r_v.^2);
    E_curr = squeeze(Ubar(4, :, :));
    E_curr(mask_p) = E_corr(mask_p);
    Ubar(4, :, :) = E_curr;
end

% Boundary states at t + dt/2
Ubar_xl = Ubar - 0.5 .* delta_x;
Ubar_xr = Ubar + 0.5 .* delta_x;
Ubar_yb = Ubar - 0.5 .* delta_y;
Ubar_yt = Ubar + 0.5 .* delta_y;

% 3. HLLC Interface fluxes
% X-interfaces (nx + 1 interfaces)
u_l_x = Ubar_xr(:, 2:ny+1, 1:nx+1);
u_r_x = Ubar_xl(:, 2:ny+1, 2:nx+2);

flux_x = zeros(4, ny, nx + 1);
for j = 1:ny
    flx_j = hllcFluxSlice(squeeze(u_l_x(:, j, :)), squeeze(u_r_x(:, j, :)), gamma(j+1,1:end-1), gamma(j+1,2:end));
    flux_x(:, j, :) = reshape(flx_j, [4, 1, nx + 1]);
end

% Y-interfaces (ny + 1 interfaces)
u_b_y = Ubar_yt(:, 1:ny+1, 2:nx+1);
u_t_y = Ubar_yb(:, 2:ny+2, 2:nx+1);

% Rotate Y: swap normal (v) and tangential (u)
u_b_rot = u_b_y([1, 3, 2, 4], :, :);
u_t_rot = u_t_y([1, 3, 2, 4], :, :);

flux_y_rot = zeros(4, ny + 1, nx);
for i = 1:nx
    fly_i = hllcFluxSlice(squeeze(u_b_rot(:, :, i)), squeeze(u_t_rot(:, :, i)), gamma(1:end-1,i+1)', gamma(2:end,i+1)');
    flux_y_rot(:, :, i) = reshape(fly_i, [4, ny + 1, 1]);
end

flux_y = flux_y_rot([1, 3, 2, 4], :, :);

% 4. Full-step conservative update
df_x = flux_x(:, :, 2:end) - flux_x(:, :, 1:end-1);
df_y = flux_y(:, 2:end, :) - flux_y(:, 1:end-1, :);

Unext = Uinner - (dt / dx) .* df_x - (dt / dy) .* df_y;

if ~physicalState2D(Unext)
    gp = squeeze(gpad(1,:,:));
    prim = uconsvToUprim2D(Upad,gp);
    sound = sqrt(gp.*squeeze(prim(4,:,:))./squeeze(prim(1,:,:)));
    ux = squeeze(prim(2,:,:)); vy = squeeze(prim(3,:,:));
    fx = eulerFluxX(Upad,gp); fy = eulerFluxY(Upad,gp);
    ax = max(abs(ux(3:end-2,2:end-2))+sound(3:end-2,2:end-2),abs(ux(3:end-2,3:end-1))+sound(3:end-2,3:end-1));
    ay = max(abs(vy(2:end-2,3:end-2))+sound(2:end-2,3:end-2),abs(vy(3:end-1,3:end-2))+sound(3:end-1,3:end-2));
    flux_x = 0.5.*(fx(:,3:end-2,2:end-2)+fx(:,3:end-2,3:end-1)-bsxfun(@times,reshape(ax,[1,ny,nx+1]),Upad(:,3:end-2,3:end-1)-Upad(:,3:end-2,2:end-2)));
    flux_y = 0.5.*(fy(:,2:end-2,3:end-2)+fy(:,3:end-1,3:end-2)-bsxfun(@times,reshape(ay,[1,ny+1,nx]),Upad(:,3:end-1,3:end-2)-Upad(:,2:end-2,3:end-2)));
    Unext = Uinner-dt/dx.*diff(flux_x,1,3)-dt/dy.*diff(flux_y,1,2);
end
if ~physicalState2D(Unext), error('Nonphysical hydrodynamic update'); end

end

function fx = eulerFluxX(U, gamma)
gamma = reshape(gamma, [1,size(U,2),size(U,3)]);
rho = max(U(1, :, :), 1e-12);
u = U(2, :, :) ./ rho;
v = U(3, :, :) ./ rho;
p = max((gamma - 1.0) .* (U(4, :, :) - 0.5 .* rho .* (u.^2 + v.^2)), 1e-12);
fx = zeros(size(U));
fx(1, :, :) = rho .* u;
fx(2, :, :) = rho .* u.^2 + p;
fx(3, :, :) = rho .* u .* v;
fx(4, :, :) = u .* (U(4, :, :) + p);
end

function fy = eulerFluxY(U, gamma)
gamma = reshape(gamma, [1,size(U,2),size(U,3)]);
rho = max(U(1, :, :), 1e-12);
u = U(2, :, :) ./ rho;
v = U(3, :, :) ./ rho;
p = max((gamma - 1.0) .* (U(4, :, :) - 0.5 .* rho .* (u.^2 + v.^2)), 1e-12);
fy = zeros(size(U));
fy(1, :, :) = rho .* v;
fy(2, :, :) = rho .* u .* v;
fy(3, :, :) = rho .* v.^2 + p;
fy(4, :, :) = v .* (U(4, :, :) + p);
end


% =========================================================================
% REACTION STEP (2D)
% =========================================================================

function [U_out, lambda_next] = reactionStep2D(U, lambda, dt_chem)
shape = size(lambda);
flat = reshape(U,4,[]);
normal = [flat(1,:); flat(2,:); flat(4,:) - 0.5.*flat(3,:).^2./flat(1,:)];
[lam, reacted] = torchReaction(normal, lambda(:)', dt_chem);
U_out = U;
U_out(4,:,:) = reshape(flat(4,:) + reacted(3,:) - normal(3,:), [1,shape]);
lambda_next = reshape(lam,shape);
end


function dt = computeCflDt2D(Uprim, dx, dy, cfl, gamma)
rho = max(squeeze(Uprim(1, :, :)), 1e-12);
u   = squeeze(Uprim(2, :, :));
v   = squeeze(Uprim(3, :, :));
p   = max(squeeze(Uprim(4, :, :)), 1e-12);

a = sqrt(gamma .* p ./ rho);
speed_x = max((abs(u(:)) + a(:)) ./ dx);
speed_y = max((abs(v(:)) + a(:)) ./ dy);

total_speed = speed_x + speed_y;
if total_speed <= 1e-12
    dt = 1e-3;
else
    dt = cfl / total_speed;
end
end

function qnext = advectScalar2D(q, old, new, fx, fy, dt_dx, dt_dy, bc_x, bc_y)
[ny,nx] = size(q);
scalar_y = bc_y;
if strcmpi(bc_y,'Inflow'), scalar_y = 'Open'; end
pad = applyBc2D(repmat(reshape(q,[1,ny,nx]),[4,1,1]),bc_x,scalar_y);
if strcmpi(bc_y,'Inflow'), pad(1,1:2,:) = 1; end
pad = squeeze(pad(1,:,:));
mx = squeeze(fx(1,:,:)); my = squeeze(fy(1,:,:));
ql = pad(3:end-2,2:end-2); qr = pad(3:end-2,3:end-1);
qb = pad(2:end-2,3:end-2); qt = pad(3:end-1,3:end-2);
qx = ql; qx(mx<0) = qr(mx<0);
qy = qb; qy(my<0) = qt(my<0);
rhoq = squeeze(old(1,:,:)).*q - dt_dx.*diff(mx.*qx,1,2) - dt_dy.*diff(my.*qy,1,1);
qnext = max(0,min(1,rhoq ./ max(squeeze(new(1,:,:)),1e-12)));
end

function valid = physicalState2D(U)
rho = squeeze(U(1,:,:));
internal = squeeze(U(4,:,:)) - 0.5.*(squeeze(U(2,:,:)).^2+squeeze(U(3,:,:)).^2)./rho;
valid = all(isfinite(U(:))) && all(rho(:)>1e-12) && all(internal(:)>1e-12);
end

function dt = paddedCfl(U,gamma,bc_x,bc_y,inlet,dx,dy,cfl)
[~,ny,nx] = size(U);
Upad = applyBc2D(U,bc_x,bc_y,inlet);
scalar_y = bc_y;
if strcmpi(bc_y,'Inflow'), scalar_y = 'Open'; end
gstate = repmat(reshape(gamma.*ones(ny,nx),[1,ny,nx]),[4,1,1]);
gpad = applyBc2D(gstate,bc_x,scalar_y);
gp = squeeze(gpad(1,:,:));
if strcmpi(bc_y,'Inflow'), gp(1:2,:) = 1.4; end
dt = computeCflDt2D(uconsvToUprim2D(Upad,gp),dx,dy,cfl,gp);
end
