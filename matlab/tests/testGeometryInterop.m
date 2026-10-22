function testGeometryInterop(folder)
% Resume 24 metric/reactive and 3 reservoir fixtures produced by Python.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
keys=strsplit(strtrim(fileread(fullfile(folder,'cases.txt'))),'\n');
for i=1:numel(keys)
    key=keys{i};s=torchReadCheckpoint(fullfile(folder,['python_' key '.h5']));
    partial=torchRunSession(s,'step_limit',1,'checkpoint',fullfile(folder,['matlab_' key '.h5']));
    final=torchRunSession(partial,'output_prefix',fullfile(folder,['history_' key]),'output_every',2);
    torchWriteCheckpoint(fullfile(folder,['matlab_final_' key '.h5']),final);
    extension='.vtr';if ~strcmp(final.config.geometry,'cartesian'),extension='.vts';end
    torchWriteVTK(fullfile(folder,['matlab_' key extension]),final);torchWriteXDMF(fullfile(folder,['matlab_' key '.xdmf']),final);
    expected=torchReadCheckpoint(fullfile(folder,['expected_' key '.h5']));
    assert(all(abs(final.state(:)-expected.state(:))<=2e-8+3e-10*abs(expected.state(:))),['Metric continuation mismatch: ' key]);
end
legacy=torchReadCheckpoint(fullfile(folder,'legacy_schema1.h5'));
legacy=torchRunSession(legacy,'checkpoint',fullfile(folder,'matlab_legacy.h5'));
expected=torchReadCheckpoint(fullfile(folder,'expected_legacy.h5'));
assert(all(abs(legacy.state(:)-expected.state(:))<1e-12+2e-11*abs(expected.state(:))),'Legacy schema-1 continuation mismatch');
fprintf('PASS 27 MATLAB geometry/boundary checkpoint continuations\n');
end
