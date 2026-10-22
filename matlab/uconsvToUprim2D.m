function Uprim = uconsvToUprim2D(Uconsv, gamma)
% UCONSVTOUPRIM2D Converts 2D conservative variables to primitive variables.
%
%   Uconsv: [4 x Ny x Nx] array [rho; rho*u; rho*v; E]
%   gamma:  ratio of specific heats (scalar or [Ny x Nx] matrix)
%   Uprim:  [4 x Ny x Nx] array [rho; u; v; p]

if nargin < 2
    gamma = 1.4;
end

rho = max(squeeze(Uconsv(1, :, :)), 1e-12);
rhou = squeeze(Uconsv(2, :, :));
rhov = squeeze(Uconsv(3, :, :));
E    = squeeze(Uconsv(4, :, :));

u = rhou ./ rho;
v = rhov ./ rho;

kinetic = 0.5 .* rho .* (u.^2 + v.^2);
internal = E - kinetic;

if isscalar(gamma)
    p = max((gamma - 1.0) .* internal, 1e-12);
else
    p = max((gamma - 1.0) .* internal, 1e-12);
end

[ny, nx] = size(rho);
Uprim = zeros(4, ny, nx);
Uprim(1, :, :) = rho;
Uprim(2, :, :) = u;
Uprim(3, :, :) = v;
Uprim(4, :, :) = p;
end
