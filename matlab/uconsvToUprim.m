function [Uprop] = uconsvToUprim(U_0, gamma)
% UCONSVTOUPRIM Converts conservative state [rho; rho*u; E] to primitive [rho; u; p].
% Works with scalar or spatially-varying gamma. Enforces positivity floors.

rho = max(U_0(1, :), 1e-12);
u   = U_0(2, :) ./ rho;
p   = max((U_0(3, :) - 0.5 .* rho .* (u.^2)) .* (gamma - 1.0), 1e-12);

Uprop = [rho; u; p];

end
