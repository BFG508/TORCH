function [U_evolution, t] = inputLoopResult(l, n, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma, C_Cfl, t_corte, Limiter, w_controler, HLLC_Variante, WaveSpeed_CalcOption, Reflective_or_Open, Div)
% INPUTLOOPRESULT Driver function for inert 1D Euler simulation.
% Calls the unified torchSolver engine.

[U_evolution, t] = torchSolver('inert', l, n, rho_l, u_l, p_l, rho_r, u_r, p_r, ...
                               gamma, C_Cfl, t_corte, Limiter, w_controler, ...
                               HLLC_Variante, WaveSpeed_CalcOption, Reflective_or_Open, Div);

end
