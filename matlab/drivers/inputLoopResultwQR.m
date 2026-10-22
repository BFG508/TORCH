function [U_evolution, t, lambda_evolution] = inputLoopResultwQR(l, n, rho_l, u_l, p_l, rho_r, u_r, p_r, lambda, C_Cfl, t_corte, Limiter, w_controler, HLLC_Variante, WaveSpeed_CalcOption, Reflective_or_Open, Divider)
% INPUTLOOPRESULTWQR Driver function for reactive 1D Euler simulation with Arrhenius chemistry.
% Calls the unified torchSolver engine.

[U_evolution, t, lambda_evolution] = torchSolver('reactive', l, n, rho_l, u_l, p_l, rho_r, u_r, p_r, ...
                                                 lambda, C_Cfl, t_corte, Limiter, w_controler, ...
                                                 HLLC_Variante, WaveSpeed_CalcOption, Reflective_or_Open, Divider);

end
