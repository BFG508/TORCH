function [U_evo, t_arr, lambda_evo] = torchSolver(varargin)
% TORCHSOLVER Unified 1D Finite-Volume CFD Solver for Compressible Reactive Gas Dynamics.
%
% Solves the 1D unsteady Euler equations with 2nd-order MUSCL spatial reconstruction,
% TVD slope limiters, HLLC approximate Riemann solver, and optional Arrhenius chemistry.
%
% Calling syntaxes:
%   1) Non-reactive (inert):
%      [U_evo, t_arr] = torchSolver('inert', l, n, rho_l, u_l, p_l, rho_r, u_r, p_r, ...
%                                   gamma, c_cfl, t_end, limiter, w_ctrl, hllc_var, ws_opt, bc_opt, div)
%   2) Reactive (H2/O2 combustion):
%      [U_evo, t_arr, lambda_evo] = torchSolver('reactive', l, n, rho_l, u_l, p_l, rho_r, u_r, p_r, ...
%                                               lambda_init, c_cfl, t_end, limiter, w_ctrl, hllc_var, ws_opt, bc_opt, div)

if nargin>0 && any(strcmp(varargin{1},{'session_dt','session_step'}))
    U=varargin{2}; gamma=varargin{3}; h=varargin{4}; c=varargin{5};
    bc=c.boundaries_xyz{1}; Up=bcCreator(U,bc);
    gp=gamma; if ~isscalar(gamma), gp=scalarBc(gamma,bc); end
    [~,~,~,waves]=waveSpeedEstimate(Up,gp,c.wavespeed);
    if strcmp(varargin{1},'session_dt')
        prim=uconsvToUprim(Up,gp);
        U_evo=min(deltaTEstimate(waves,h(1),c.cfl),c.cfl*h(1)/max(abs(prim(2,:))+soundSpeedCalc(prim,gp)));
        t_arr=[];
    else
        dt=varargin{6}; reconstructed=musclApplication(Up,dt,h(1),gp,c.cfl,c.limiter,c.controller,bc);
        workers=1;if isfield(c,'runtime_workers'),workers=c.runtime_workers;end
        F=hllcFluxCalc(reconstructed,waves,gp,c.variant,workers);
        [U_evo,F]=safeUpdate(Up,F,dt,h(1),gp); t_arr={F};
    end
    lambda_evo=[]; return
end

if nargin ~= 18
    error('torchSolver requires exactly 18 simulation arguments.');
end

mode = varargin{1};
l       = varargin{2};
n       = varargin{3};
rho_l   = varargin{4};
u_l     = varargin{5};
p_l     = varargin{6};
rho_r   = varargin{7};
u_r     = varargin{8};
p_r     = varargin{9};

assert(any(strcmpi(mode, {'inert','reactive'})), 'Unknown simulation mode');
is_reactive = strcmpi(mode, 'reactive');

if is_reactive
    lambda_init = varargin{10};
    c_cfl       = varargin{11};
    t_end       = varargin{12};
    limiter     = varargin{13};
    w_ctrl      = varargin{14};
    hllc_var    = varargin{15};
    ws_opt      = varargin{16};
    bc_opt      = varargin{17};
    div         = varargin{18};
    gamma       = gammaCalc(lambda_init);
else
    gamma       = varargin{10};
    c_cfl       = varargin{11};
    t_end       = varargin{12};
    limiter     = varargin{13};
    w_ctrl      = varargin{14};
    hllc_var    = varargin{15};
    ws_opt      = varargin{16};
    bc_opt      = varargin{17};
    div         = varargin{18};
    lambda_init = [];
end

assert(all(isfinite([l,n,c_cfl,t_end])) && l > 0 && n >= 2 && n == fix(n) && c_cfl > 0 && c_cfl <= 0.5 && t_end >= 0, 'Invalid domain, grid, CFL or final time');
assert(all(isfinite([rho_l,u_l,p_l,rho_r,u_r,p_r])) && all([rho_l,p_l,rho_r,p_r]>0), 'Invalid initial primitives');
assert(isscalar(div) && isfinite(div) && div>0 && div<100 && div~=1, 'Invalid divider');
if is_reactive
    assert(isrow(lambda_init) && numel(lambda_init)==n && all(isfinite(lambda_init)) && all(lambda_init>=0 & lambda_init<=1), 'Invalid initial reactant fraction');
else
    assert(isscalar(gamma) && isfinite(gamma) && gamma>1, 'Invalid gamma');
end
dx = l / n;

% Initialize state vector [rho; rho*u; E]
if is_reactive
    U_curr = stateInit(n, rho_l, u_l, p_l, rho_r, u_r, p_r, lambda_init, div);
    lambda_curr = lambda_init;
    lambda_evo  = lambda_init;
else
    U_curr = stateInit(n, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma, div);
    lambda_curr = [];
    lambda_evo  = [];
end

assert(physicalState(U_curr), 'Nonphysical initial conservative state');
U_evo = U_curr;
t_arr = 0.0;

% Time integration loop
while t_arr(end) < t_end

    % 0 - Ghost cell boundary conditions
    U_bc = bcCreator(U_curr, bc_opt);

    if is_reactive
        lambda_bc = scalarBc(lambda_curr, bc_opt);
        gamma_bc  = gammaCalc(lambda_bc);
        gamma_arg = gamma_bc;
    else
        gamma_arg = gamma;
    end

    % 1 - Characteristic wave speeds (S_L, S_*, S_R)
    [~, ~, ~, ws_mat] = waveSpeedEstimate(U_bc, gamma_arg, ws_opt);

    % 2 - Adaptive CFL time step
    prim_dt = uconsvToUprim(U_bc,gamma_arg);
    dt = min(deltaTEstimate(ws_mat,dx,c_cfl), c_cfl*dx/max(abs(prim_dt(2,:))+soundSpeedCalc(prim_dt,gamma_arg)));
    if t_arr(end) + dt > t_end
        dt = t_end - t_arr(end);
    end
    if ~isfinite(dt) || dt <= 1e-15
        error('Time integration stalled before requested final time');
    end

    if is_reactive
        % Formal 2nd-order Strang operator splitting: R(dt/2) -> H(dt) -> R(dt/2)
        cfl_stable = false;
        for retry = 1:60
            dt_r_half = 0.5 * dt;
            [lambda_half, U_chem_1] = kineticsCalc(lambda_curr, dt_r_half, U_curr);
            gp = gammaCalc(lambda_half);
            prim_dt = uconsvToUprim(U_chem_1,gp);
            allowed = c_cfl*dx/max(abs(prim_dt(2,:))+soundSpeedCalc(prim_dt,gp));
            if dt <= allowed*(1+1e-12), cfl_stable = true; break; end
            dt = min(0.5*dt,allowed);
        end
        if ~cfl_stable, error('Could not find a CFL-stable reactive step'); end

        % Step 2: Hydrodynamics full-step (dt)
        U_chem_1_bc = bcCreator(U_chem_1, bc_opt);
        lambda_half_bc = scalarBc(lambda_half, bc_opt);
        gamma_half_bc  = gammaCalc(lambda_half_bc);

        [~, ~, ~, ws_mat_step] = waveSpeedEstimate(U_chem_1_bc, gamma_half_bc, ws_opt);
        U_muscl = musclApplication(U_chem_1_bc, dt, dx, gamma_half_bc, c_cfl, limiter, w_ctrl, bc_opt);
        f_hllc  = hllcFluxCalc(U_muscl, ws_mat_step, gamma_half_bc, hllc_var);
        [U_hydro, f_hllc] = safeUpdate(U_chem_1_bc, f_hllc, dt, dx, gamma_half_bc);

        lambda_half = advectScalar(lambda_half, U_chem_1, U_hydro, f_hllc(1,:), dt/dx, bc_opt);

        % Step 3: Chemistry second half-step (dt/2)
        [lambda_next, U_next] = kineticsCalc(lambda_half, dt_r_half, U_hydro);

        lambda_curr = lambda_next;
        lambda_evo  = cat(1, lambda_evo, lambda_next);
    else
        % 3 - MUSCL predictor
        U_muscl = musclApplication(U_bc, dt, dx, gamma_arg, c_cfl, limiter, w_ctrl, bc_opt);

        % 4 - HLLC flux and hydro advance
        f_hllc = hllcFluxCalc(U_muscl, ws_mat, gamma_arg, hllc_var);
        [U_next, f_hllc] = safeUpdate(U_bc, f_hllc, dt, dx, gamma_arg);
    end

    % Accumulate evolution
    U_evo = cat(3, U_evo, U_next);
    t_arr = [t_arr, t_arr(end) + dt];
    U_curr = U_next;

end

end


% =========================================================================
% INTERNAL SUBFUNCTIONS
% =========================================================================

function [U_0] = stateInit(n, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma_or_lambda, div)
    if div <= 1.0
        frac = div;
    else
        frac = div / 100.0;
    end
    m_i = max(1, min(n - 1, round(frac * n)));

    left_prim  = repmat([rho_l; u_l; p_l], 1, m_i);
    right_prim = repmat([rho_r; u_r; p_r], 1, n - m_i);
    U_prim     = [left_prim, right_prim];

    U_0 = uprimToUconsv(U_prim, gamma_or_lambda);
end


function [U_bc] = bcCreator(U_in, bc_opt)
    if strcmpi(bc_opt, 'Reflective')
        bc_l = [ U_in(1, 1);   -U_in(2, 1);   U_in(3, 1)];
        bc_r = [ U_in(1, end); -U_in(2, end); U_in(3, end)];
    elseif strcmpi(bc_opt, 'Periodic')
        bc_l = U_in(:, end);
        bc_r = U_in(:, 1);
    elseif strcmpi(bc_opt, 'Open')
        bc_l = U_in(:, 1);
        bc_r = U_in(:, end);
    else
        error('Unknown boundary condition: %s', bc_opt);
    end
    U_bc = [bc_l, U_in, bc_r];
end


function [S_L, S_s, S_R, ws_mat] = waveSpeedEstimate(U_bc, gamma_in, ws_opt)
    if all(gamma_in <= 1.05)
        gamma = gammaCalc(gamma_in);
    else
        gamma = gamma_in;
    end

    U_prim = uconsvToUprim(U_bc, gamma);

    if isscalar(gamma)
        g_l = gamma; g_r = gamma;
    else
        g_l = gamma(1:end-1); g_r = gamma(2:end);
    end

    U_L = U_prim(:, 1:end-1);
    rho_L = max(U_L(1, :), 1e-12);
    u_L   = U_L(2, :);
    p_L   = max(U_L(3, :), 0.0);
    a_L   = soundSpeedCalc(U_L, g_l);
    e_L   = p_L ./ ((g_l - 1.0) .* rho_L);
    E_L   = rho_L .* (0.5 .* (u_L.^2) + e_L);

    U_R = U_prim(:, 2:end);
    rho_R = max(U_R(1, :), 1e-12);
    u_R   = U_R(2, :);
    p_R   = max(U_R(3, :), 0.0);
    a_R   = soundSpeedCalc(U_R, g_r);
    e_R   = p_R ./ ((g_r - 1.0) .* rho_R);
    E_R   = rho_R .* (0.5 .* (u_R.^2) + e_R);

    if strcmpi(ws_opt, 'Pressure Based')
        p_s = pressureEstimate(U_bc, gamma);

        q_L = ones(1, length(p_s));
        mask_l = p_s > p_L;
        if isscalar(gamma)
            gl_sub = gamma;
        else
            gl_sub = g_l(mask_l);
        end
        q_L(mask_l) = sqrt(1.0 + (gl_sub + 1.0) ./ (2.0 .* gl_sub) .* (p_s(mask_l) ./ p_L(mask_l) - 1.0));

        q_R = ones(1, length(p_s));
        mask_r = p_s > p_R;
        if isscalar(gamma)
            gr_sub = gamma;
        else
            gr_sub = g_r(mask_r);
        end
        q_R(mask_r) = sqrt(1.0 + (gr_sub + 1.0) ./ (2.0 .* gr_sub) .* (p_s(mask_r) ./ p_R(mask_r) - 1.0));

        S_L = u_L - a_L .* q_L;
        S_R = u_R + a_R .* q_R;

    elseif strcmpi(ws_opt, 'Enthalpy Based')
        sqrt_rho_l = sqrt(rho_L);
        sqrt_rho_r = sqrt(rho_R);
        denom_roe = max(sqrt_rho_l + sqrt_rho_r, 1e-12);

        u_avg = (sqrt_rho_l .* u_L + sqrt_rho_r .* u_R) ./ denom_roe;

        H_L = (E_L + p_L) ./ rho_L;
        H_R = (E_R + p_R) ./ rho_R;
        H_avg = (sqrt_rho_l .* H_L + sqrt_rho_r .* H_R) ./ denom_roe;

        if isscalar(gamma)
            gamma_avg = gamma;
        else
            gamma_avg = 0.5 * (g_l + g_r);
        end
        a_avg_sq = (gamma_avg - 1.0) .* (H_avg - 0.5 .* (u_avg.^2));
        a_avg = sqrt(max(a_avg_sq, 1e-10));

        % Einfeldt positivity-preserving wave speed bounds
        S_L = min(u_L - a_L, u_avg - a_avg);
        S_R = max(u_R + a_R, u_avg + a_avg);
    else
        error('Unknown wave speed option: %s', ws_opt);
    end

    denom_s_s = rho_L .* (S_L - u_L) - rho_R .* (S_R - u_R);
    denom_s_s(abs(denom_s_s) < 1e-14) = 1e-14;

    S_s = (p_R - p_L + rho_L .* u_L .* (S_L - u_L) - rho_R .* u_R .* (S_R - u_R)) ./ denom_s_s;
    ws_mat = [S_L; S_s; S_R];
end


function [p_s] = pressureEstimate(U_bc, gamma)
    U_prim = uconsvToUprim(U_bc, gamma);
    U_L = U_prim(:, 1:end-1);
    U_R = U_prim(:, 2:end);

    if isscalar(gamma)
        g_l = gamma; g_r = gamma;
    else
        g_l = gamma(1:end-1); g_r = gamma(2:end);
    end

    rho_L = max(U_L(1, :), 1e-12);
    u_L   = U_L(2, :);
    p_L   = max(U_L(3, :), 0.0);
    a_L   = soundSpeedCalc(U_L, g_l);

    rho_R = max(U_R(1, :), 1e-12);
    u_R   = U_R(2, :);
    p_R   = max(U_R(3, :), 0.0);
    a_R   = soundSpeedCalc(U_R, g_r);

    rho_avg = 0.5 * (rho_R + rho_L);
    a_avg   = 0.5 * (a_L + a_R);

    p_pvrs = 0.5 * (p_L + p_R) - 0.5 * (u_R - u_L) .* rho_avg .* a_avg;
    p_s    = max(0.0, p_pvrs);
end


function [dt] = deltaTEstimate(ws_mat, dx, c_cfl)
    max_ws = max(abs(ws_mat(:)));
    if max_ws <= 0 || ~isfinite(max_ws)
        max_ws = 1e-6;
    end
    dt = (c_cfl * dx) / max_ws;
end


function [U_muscl] = musclApplication(U_bc, dt, dx, gamma_arg, c_cfl, limiter, w_ctrl, bc_opt)
    if all(gamma_arg <= 1.05)
        gamma = gammaCalc(gamma_arg);
    else
        gamma = gamma_arg;
    end

    U_prim = uconsvToUprim(U_bc, gamma);
    a = soundSpeedCalc(U_prim, gamma);
    u = U_prim(2, :);

    % Upwind / downwind differences
    delta_i_uw = U_bc(:, 2:end-1) - U_bc(:, 1:end-2);
    delta_i_dw = U_bc(:, 3:end)   - U_bc(:, 2:end-1);
    n_cells = size(delta_i_uw, 2);

    switch w_ctrl
        case '1'
            w = ones(1, n_cells);
        case '-1'
            w = -ones(1, n_cells);
        case '0.5'
            w = 0.5 * ones(1, n_cells);
        case 'Sound Speed Controlled'
            mach = abs(u(2:end-1) ./ max(a(2:end-1), 1e-12));
            mach(~isfinite(mach)) = 0.0;
            w = (1.0 - mach) ./ (1.0 + mach);
            w = max(-1.0, min(1.0, w));
        otherwise
            w = zeros(1, n_cells);
    end

    w_mat = repmat(w, 3, 1);
    delta_i = 0.5 .* (delta_i_uw + delta_i_dw);

    % TVD slope limiter
    delta_lim = slopeLimiterCalc(delta_i, c_cfl, delta_i_uw, delta_i_dw, w, limiter);

    % MUSCL reconstruction with 0.5t predictor
    U_inner = U_bc(:, 2:end-1);
    U_L = U_inner - 0.5 .* delta_lim;
    U_R = U_inner + 0.5 .* delta_lim;

    if isscalar(gamma)
        g_inner = gamma;
    else
        g_inner = gamma(2:end-1);
    end

    d_fu = 0.5 * (dt / dx) .* (uToFU(U_L, g_inner) - uToFU(U_R, g_inner));
    U_L_dt = U_L + d_fu;
    U_R_dt = U_R + d_fu;

    pL = (g_inner-1) .* (U_L_dt(3,:) - 0.5 .* U_L_dt(2,:).^2 ./ max(U_L_dt(1,:),1e-12));
    pR = (g_inner-1) .* (U_R_dt(3,:) - 0.5 .* U_R_dt(2,:).^2 ./ max(U_R_dt(1,:),1e-12));
    badL = U_L_dt(1,:) <= 1e-12 | pL <= 1e-12 | any(~isfinite(U_L_dt),1);
    badR = U_R_dt(1,:) <= 1e-12 | pR <= 1e-12 | any(~isfinite(U_R_dt),1);
    U_L_dt(:,badL) = U_inner(:,badL);
    U_R_dt(:,badR) = U_inner(:,badR);
    U_sub = zeros(3, 2 * n_cells);
    U_sub(:, 1:2:end) = U_L_dt;
    U_sub(:, 2:2:end) = U_R_dt;

    U_muscl = bcCreator(U_sub, bc_opt);
end


function [delta_lim] = slopeLimiterCalc(delta_i, c_cfl, di_uw, di_dw, w, limiter)
    r = nanToZero(di_uw ./ di_dw);

    if iscolumn(w)
        w_mat = repmat(w', size(delta_i, 1), 1);
    elseif isrow(w)
        w_mat = repmat(w, size(delta_i, 1), 1);
    else
        w_mat = w;
    end

    denom = 1.0 - w_mat + (1.0 + w_mat) .* r;
    denom(abs(denom) < 1e-12) = 1e-12;
    eps_r = max(0.0, nanToZero(2.0 ./ denom));

    eps = zeros(size(delta_i));

    switch limiter
        case 'VanLeer'
            pos = r > 0;
            phi = nanToZero((2.0 .* r) ./ (1.0 + r));
            eps(pos) = min(phi(pos), eps_r(pos));
        case 'Albada'
            pos = r > 0;
            phi = nanToZero((r .* (1.0 + r)) ./ (1.0 + r.^2));
            eps(pos) = min(phi(pos), eps_r(pos));
        case 'MINBEE'
            % Authentic Roe MINBEE: r for 0 < r < 1, 1 for r >= 1
            pos1 = (r > 0) & (r < 1.0);
            pos2 = r >= 1.0;
            eps(pos1) = r(pos1);
            eps(pos2) = min(1.0, eps_r(pos2));
        case 'Superbee'
            pos = r > 0;
            term1 = min(2.0 .* r, 1.0);
            term2 = min(r, 2.0);
            phi = max(0.0, max(term1, term2));
            eps(pos) = min(phi(pos), eps_r(pos));
        case 'Unos'
            eps = min(max(0.0, min(min(2.0, 2.0 .* r), (2.0 + r) ./ 3.0)), eps_r);
        otherwise
            pos = r > 0;
            phi = nanToZero((2.0 .* r) ./ (1.0 + r));
            eps(pos) = min(phi(pos), eps_r(pos));
    end

    eps = max(0.0, nanToZero(eps));
    slope = eps .* delta_i;
    bound = 2 .* min(abs(di_uw), abs(di_dw));
    delta_lim = sign(di_dw) .* min(abs(slope), bound);
    delta_lim(di_uw .* di_dw <= 0) = 0;
end


function [F_hllc] = hllcFluxCalc(U_muscl, ws_mat, gamma_arg, variant,workers)
    if nargin<5,workers=1;end
    if workers>1
        F_hllc=torchParallelLegacy1D(@hllcFluxCalc,U_muscl,ws_mat,gamma_arg,variant,workers);return;
    end
    l1 = size(ws_mat, 2);

    if all(gamma_arg <= 1.05)
        gamma = gammaCalc(gamma_arg);
    else
        gamma = gamma_arg;
    end

    if isscalar(gamma)
        g_l = gamma; g_r = gamma;
    else
        g_l = gamma(1:end-1); g_r = gamma(2:end);
    end

    U_L = U_muscl(:, 1:2:2*l1);
    U_R = U_muscl(:, 2:2:2*l1);

    U_prim_L = uconsvToUprim(U_L, g_l);
    rho_L = max(U_prim_L(1, :), 1e-12);
    u_L   = U_prim_L(2, :);
    p_L   = max(U_prim_L(3, :), 0.0);
    e_L   = p_L ./ ((g_l - 1.0) .* rho_L);
    E_L   = rho_L .* (0.5 .* (u_L.^2) + e_L);

    U_prim_R = uconsvToUprim(U_R, g_r);
    rho_R = max(U_prim_R(1, :), 1e-12);
    u_R   = U_prim_R(2, :);
    p_R   = max(U_prim_R(3, :), 0.0);
    e_R   = p_R ./ ((g_r - 1.0) .* rho_R);
    E_R   = rho_R .* (0.5 .* (u_R.^2) + e_R);

    F_L = uToFU(U_L, g_l);
    F_R = uToFU(U_R, g_r);

    S_L = ws_mat(1, :);
    S_R = ws_mat(3, :);
    % Contact speed must use the MUSCL-predicted interface states.
    % A cell-average contact speed leaves acoustic transport first order.
    denominator=rho_L.*(S_L-u_L)-rho_R.*(S_R-u_R);
    denominator(abs(denominator)<1e-14)=1e-14;
    S_s=(p_R-p_L+rho_L.*u_L.*(S_L-u_L)-rho_R.*u_R.*(S_R-u_R))./denominator;
    S_s=max(S_L,min(S_R,S_s));

    eps_val = 1e-12;

    denom_sl_ss = S_L - S_s;
    denom_sr_ss = S_R - S_s;

    denom_sl_u = S_L - u_L;
    denom_sl_u(abs(denom_sl_u) < eps_val) = -eps_val;
    e_star_L = E_L ./ rho_L + (S_s - u_L) .* (S_s + p_L ./ (rho_L .* denom_sl_u));

    denom_sr_u = S_R - u_R;
    denom_sr_u(abs(denom_sr_u) < eps_val) = eps_val;
    e_star_R = E_R ./ rho_R + (S_s - u_R) .* (S_s + p_R ./ (rho_R .* denom_sr_u));

    valid_l = abs(denom_sl_ss) > eps_val;
    factor_L = zeros(1, l1);
    factor_L(valid_l) = rho_L(valid_l) .* ((S_L(valid_l) - u_L(valid_l)) ./ denom_sl_ss(valid_l));
    U_Ls = [factor_L; factor_L .* S_s; factor_L .* e_star_L];
    U_Ls(:, ~valid_l) = U_L(:, ~valid_l);

    valid_r = abs(denom_sr_ss) > eps_val;
    factor_R = zeros(1, l1);
    factor_R(valid_r) = rho_R(valid_r) .* ((S_R(valid_r) - u_R(valid_r)) ./ denom_sr_ss(valid_r));
    U_Rs = [factor_R; factor_R .* S_s; factor_R .* e_star_R];
    U_Rs(:, ~valid_r) = U_R(:, ~valid_r);

    F_Ls = F_L + S_L .* (U_Ls - U_L);
    F_Rs = F_R + S_R .* (U_Rs - U_R);

    F_hllc = zeros(3, l1);
    m_l   = S_L > 0;
    m_sl  = (S_L <= 0) & (S_s > 0);
    m_sr  = (S_s <= 0) & (S_R >= 0);
    m_r   = S_R < 0;

    F_hllc(:, m_l)  = F_L(:, m_l);
    F_hllc(:, m_sl) = F_Ls(:, m_sl);
    F_hllc(:, m_sr) = F_Rs(:, m_sr);
    F_hllc(:, m_r)  = F_R(:, m_r);
end


function [U_next] = fluxApplication(U_bc, F_flux, dt, dx)
    d_flux = F_flux(:, 2:end) - F_flux(:, 1:end-1);
    dU = -(dt / dx) .* d_flux;
    U_next = U_bc(:, 2:end-1) + dU;

end


function [lambda_np1, U_out] = kineticsCalc(lambda_n, dt_r, U_in)
    [lambda_np1, U_out] = torchReaction(U_in, lambda_n, dt_r);
end


function [FU] = uToFU(U, gamma)
    if nargin < 2 || isempty(gamma)
        gamma = 1.4;
    end
    U_prim = uconsvToUprim(U, gamma);
    rho = U_prim(1, :);
    u   = U_prim(2, :);
    p   = U_prim(3, :);
    FU = [rho .* u; rho .* (u.^2) + p; u .* (U(3, :) + p)];
end


function [gamma] = gammaCalc(lambda)
    gamma = (4.5 - lambda) ./ (3.5 - lambda);
end


function [a] = soundSpeedCalc(U_prim, gamma)
    rho = max(U_prim(1, :), 1e-12);
    p   = max(U_prim(3, :), 0.0);
    a   = sqrt(gamma .* p ./ rho);
end


function [U_prim] = uconsvToUprim(U_consv, gamma)
    rho = max(U_consv(1, :), 1e-12);
    u   = U_consv(2, :) ./ rho;
    p   = max((U_consv(3, :) - 0.5 .* rho .* (u.^2)) .* (gamma - 1.0), 1e-12);
    U_prim = [rho; u; p];
end


function [U_consv] = uprimToUconsv(U_prim, gamma_or_lambda)
    if nargin < 2 || isempty(gamma_or_lambda)
        gamma = 1.4;
    elseif all(gamma_or_lambda <= 1.05)
        gamma = gammaCalc(gamma_or_lambda);
    else
        gamma = gamma_or_lambda;
    end
    rho = U_prim(1, :);
    u   = U_prim(2, :);
    p   = max(U_prim(3, :), 0.0);
    E   = rho .* (0.5 .* (u.^2) + p ./ ((gamma - 1.0) .* rho));
    U_consv = [rho; rho .* u; E];
end


function [out] = nanToZero(in)
    out = in;
    out(~isfinite(out)) = 0.0;
end

function padded = scalarBc(q, bc)
if strcmpi(bc, 'Periodic')
    padded = [q(end), q, q(1)];
else
    padded = [q(1), q, q(end)];
end
end

function qnext = advectScalar(q, old, new, massFlux, dt_dx, bc)
qpad = scalarBc(q, bc);
upwind = qpad(1:end-1);
mask = massFlux < 0;
right = qpad(2:end);
upwind(mask) = right(mask);
flux = massFlux .* upwind;
qnext = max(0, min(1, (old(1,:) .* q - dt_dx .* diff(flux)) ./ max(new(1,:),1e-12)));
end

function valid = physicalState(U)
valid = all(isfinite(U(:))) && all(U(1,:)>1e-12) && all(U(3,:)-0.5.*U(2,:).^2./U(1,:)>1e-12);
end

function [next, flux] = safeUpdate(U, flux, dt, dx, gamma)
next = fluxApplication(U,flux,dt,dx);
if ~physicalState(next)
    prim = uconsvToUprim(U,gamma);
    speed = abs(prim(2,:))+soundSpeedCalc(prim,gamma);
    alpha = max(speed(1:end-1),speed(2:end));
    f = uToFU(U,gamma);
    flux = 0.5.*(f(:,1:end-1)+f(:,2:end)-bsxfun(@times,alpha,diff(U,1,2)));
    next = fluxApplication(U,flux,dt,dx);
end
if ~physicalState(next), error('Nonphysical hydrodynamic update'); end
end
