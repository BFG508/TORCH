function flux = torchHllcFlux(u_l, u_r, gamma_l, gamma_r)
if nargin < 4, gamma_r = gamma_l; end
gamma = 0.5 .* (gamma_l + gamma_r);
% Directional HLLC for four/five rows; normal momentum is row two.
rho_l = max(u_l(1, :), 1e-12);
rho_r = max(u_r(1, :), 1e-12);

un_l = u_l(2, :) ./ rho_l;
un_r = u_r(2, :) ./ rho_r;

ut_l = u_l(3:end-1, :) ./ rho_l;
ut_r = u_r(3:end-1, :) ./ rho_r;

ke_l = 0.5 .* rho_l .* (un_l.^2 + sum(ut_l.^2,1));
ke_r = 0.5 .* rho_r .* (un_r.^2 + sum(ut_r.^2,1));

p_l = max((gamma_l - 1.0) .* (u_l(end, :) - ke_l), 1e-12);
p_r = max((gamma_r - 1.0) .* (u_r(end, :) - ke_r), 1e-12);

a_l = sqrt(gamma_l .* p_l ./ rho_l);
a_r = sqrt(gamma_r .* p_r ./ rho_r);

% Physical fluxes
f_l = zeros(size(u_l));
f_l(1, :) = rho_l .* un_l;
f_l(2, :) = rho_l .* un_l.^2 + p_l;
f_l(3:end-1, :) = rho_l .* un_l .* ut_l;
f_l(end, :) = un_l .* (u_l(end, :) + p_l);

f_r = zeros(size(u_r));
f_r(1, :) = rho_r .* un_r;
f_r(2, :) = rho_r .* un_r.^2 + p_r;
f_r(3:end-1, :) = rho_r .* un_r .* ut_r;
f_r(end, :) = un_r .* (u_r(end, :) + p_r);

% Roe averages
sqrt_l = sqrt(rho_l);
sqrt_r = sqrt(rho_r);
inv_d = 1.0 ./ (sqrt_l + sqrt_r);

u_roe = (sqrt_l .* un_l + sqrt_r .* un_r) .* inv_d;
v_roe = (sqrt_l .* ut_l + sqrt_r .* ut_r) .* inv_d;
h_l   = (u_l(end, :) + p_l) ./ rho_l;
h_r   = (u_r(end, :) + p_r) ./ rho_r;
h_roe = (sqrt_l .* h_l + sqrt_r .* h_r) .* inv_d;
ke_roe = 0.5 .* (u_roe.^2 + sum(v_roe.^2,1));
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
u_star_l(3:end-1, :) = rho_sl .* ut_l;
u_star_l(end, :) = rho_sl .* (u_l(end, :) ./ rho_l + (s_star - un_l) .* (s_star + p_l ./ (rho_l .* denom_sl_un)));

u_star_r = zeros(size(u_r));
u_star_r(1, :) = rho_sr;
u_star_r(2, :) = rho_sr .* s_star;
u_star_r(3:end-1, :) = rho_sr .* ut_r;
u_star_r(end, :) = rho_sr .* (u_r(end, :) ./ rho_r + (s_star - un_r) .* (s_star + p_r ./ (rho_r .* denom_sr_un)));

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
