function testCheckpointInterop(folder)
% Read Python fixtures; continue and write snapshots for independent Python checks.
addpath(fileparts(fileparts(mfilename('fullpath'))));
setupTorch;
for dim=1:3
    for reactive=[0,1]
        key=sprintf('%d_%d',dim,reactive);
        s=torchReadCheckpoint(fullfile(folder,['python_' key '.h5']));
        partial=torchRunSession(s,'step_limit',1,'checkpoint',fullfile(folder,['matlab_' key '.h5']));
        final=torchRunSession(partial,'output_prefix',fullfile(folder,['history_' key]),'output_every',2);
        torchWriteCheckpoint(fullfile(folder,['matlab_final_' key '.h5']),final);
        torchWriteVTK(fullfile(folder,['matlab_' key '.vtr']),final);
        torchWriteXDMF(fullfile(folder,['matlab_' key '.xdmf']),final);
        expected=torchReadCheckpoint(fullfile(folder,['expected_' key '.h5']));
        error=abs(final.state(:)-expected.state(:));tolerance=1e-8+2e-11*abs(expected.state(:));
        assert(all(error<=tolerance),'Python/MATLAB continuation mismatch');
    end
end
rejected=false;try,torchReadCheckpoint(fullfile(folder,'znd.h5'));catch,rejected=true;end
assert(rejected,'Unsupported MATLAB ZND restart silently accepted');
fprintf('MATLAB/PYTHON CHECKPOINT EXCHANGE PASSED!\n');
end
