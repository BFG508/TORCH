function plotSodLocal(csvName)
% PLOTSODLOCAL Reads and plots CFD output data (x, p, rho, u) from CSV file.

if nargin < 1 || isempty(csvName)
    csvName = 'sod_data.csv';
end

scriptDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(scriptDir);
repoDir = fileparts(matlabDir);
candidate_paths = {
    csvName, ...
    fullfile(pwd, csvName), ...
    fullfile(pwd, 'out', csvName), ...
    fullfile(scriptDir, csvName), ...
    fullfile(scriptDir, 'out', csvName), ...
    fullfile(matlabDir, csvName), ...
    fullfile(matlabDir, 'out', csvName), ...
    fullfile(repoDir, 'out', csvName)
};

csvPath = '';
for k = 1:numel(candidate_paths)
    if isfile(candidate_paths{k})
        csvPath = candidate_paths{k};
        break;
    end
end

if isempty(csvPath)
    fprintf('Could not find CSV file: %s\n', csvName);
    if usejava('jvm') && usejava('awt')
        [f, p] = uigetfile({'*.csv', 'CSV files'}, 'Select CSV file');
        if isequal(f, 0); error('No CSV file selected.'); end
        csvPath = fullfile(p, f);
    else
        error('CSV file not found.');
    end
end

T = readtable(csvPath);
col_names = T.Properties.VariableNames;

if all(ismember({'x', 'p', 'rho', 'u'}, col_names))
    x = T.x; p = T.p; rho = T.rho; u = T.u;
else
    x = T{:, 1}; p = T{:, 2}; rho = T{:, 3}; u = T{:, 4};
end

figure('Name', 'TORCH CFD Results');
subplot(3, 1, 1); plot(x, p, 'b-', 'LineWidth', 1.5); grid on; ylabel('Pressure (Pa)'); title('Pressure');
subplot(3, 1, 2); plot(x, rho, 'r-', 'LineWidth', 1.5); grid on; ylabel('Density (kg/m^3)'); title('Density');
subplot(3, 1, 3); plot(x, u, 'g-', 'LineWidth', 1.5); grid on; xlabel('x (m)'); ylabel('Velocity (m/s)'); title('Velocity');

end
