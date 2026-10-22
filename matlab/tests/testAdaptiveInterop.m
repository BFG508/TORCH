function testAdaptiveInterop(folder)
% Continue Python AMR checkpoints and export portable leaf geometry/fields.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
keys=strsplit(strtrim(fileread(fullfile(folder,'cases.txt'))),'\n');
for i=1:numel(keys)
    key=keys{i};s=torchReadCheckpoint(fullfile(folder,['python_' key '.h5']));s.workers=2;
    partial=torchRunSession(s,'step_limit',1,'checkpoint',fullfile(folder,['matlab_' key '.h5']));
    final=torchRunSession(partial,'output_prefix',fullfile(folder,['history_' key]),'output_every',2);
    torchWriteCheckpoint(fullfile(folder,['matlab_final_' key '.h5']),final);torchWriteVTK(fullfile(folder,['matlab_' key '.vtu']),final);torchWriteXDMF(fullfile(folder,['matlab_' key '.xdmf']),final);
    expected=torchReadCheckpoint(fullfile(folder,['expected_' key '.h5']));
    assert(isequal(final.mesh.keys,expected.mesh.keys),['Adaptive topology mismatch: ' key]);
    assert(all(abs(final.state(:)-expected.state(:))<=3e-8+5e-10*abs(expected.state(:))),['Adaptive state mismatch: ' key]);
end
fprintf('PASS 24 MATLAB AMR cross-language continuations\n');
end
