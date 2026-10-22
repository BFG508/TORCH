function [lam, state] = torchReaction(U, lambda, dt)
% SI exponential-midpoint Arrhenius chemistry; E [J/m^3], p [Pa].
rate0 = reactionRate(U, lambda);
[lamMid, Umid] = advance(U, lambda, rate0, 0.5*dt);
[lam, state] = advance(U, lambda, reactionRate(Umid, lamMid), dt);
end

function rate = reactionRate(U, lambda)
gamma = (4.5-lambda) ./ (3.5-lambda);
prim = uconsvToUprim(U, gamma);
M = lambda .* (2/3) .* 2.016 + lambda .* (1/3) .* 32 + (1-lambda).*18.016;
T = max(prim(3,:) .* M ./ (max(prim(1,:),1e-12) .* 8314.46), 1e-12);
rate = 1.76e5 .* exp(-(17.15*4184) ./ (8.314 .* T));
end

function [lam, state] = advance(U, lambda, rate, dt)
lam = lambda .* exp(-rate .* dt);
state = U;
state(3,:) = U(3,:) + 141584000 .* U(1,:) .* (lambda-lam);
end
