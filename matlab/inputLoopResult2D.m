function [U_final, t_arr, lambda_final, X, Y, diagnostics] = inputLoopResult2D(varargin)
% INPUTLOOPRESULT2D Driver function for 2D CFD simulation (inert & reactive).
% Calls the unified torchSolver2D finite-volume engine.
%
% Syntaxes:
%   1) Using named preset:
%      [U_final, t_arr, lambda_final, X, Y, diagnostics] = inputLoopResult2D('2d_explosion', 40, 40, 0.2, 0.4, 'VanLeer');
%   2) Using explicit mode:
%      [U_final, t_arr, lambda_final, X, Y, diagnostics] = inputLoopResult2D('preset', '2d_rde_chamber', 40, 40, 0.005, 0.35, 'Superbee');
%   3) Custom initial condition:
%      [U_final, t_arr, lambda_final, X, Y, diagnostics] = inputLoopResult2D('custom', U_init, lx, ly, nx, ny, t_end, cfl, limiter, bc_x, bc_y);

if nargin == 0
    error('inputLoopResult2D requires simulation parameters.');
end

first_arg = varargin{1};
if ischar(first_arg) && ~strcmpi(first_arg, 'preset') && ~strcmpi(first_arg, 'custom')
    % Passed preset directly as first string argument
    preset_name = first_arg;
    varargin_fwd = [{'preset', preset_name}, varargin(2:end)];
    [U_final, t_arr, lambda_final, X, Y, diagnostics] = torchSolver2D(varargin_fwd{:});
else
    [U_final, t_arr, lambda_final, X, Y, diagnostics] = torchSolver2D(varargin{:});
end

end
