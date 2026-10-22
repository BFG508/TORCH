function setupTorch
% SETUPTORCH Add TORCH runtime folders to the MATLAB/Octave path.
%   addpath('/path/to/TORCH/matlab'); setupTorch;
%   Paths are resolved relative to this file, independent of the current folder.

root = fileparts(mfilename('fullpath'));
folders = {'solvers', 'physics', 'drivers', 'sessions', 'io', ...
           'visualization', 'gui', 'geometry', 'validation', 'adaptive'};
addpath(root);
for k = 1:numel(folders)
    addpath(fullfile(root, folders{k}));
end
end
