function pool = torchWorkers(workers)
% Validate CPU worker selection without deleting/replacing an existing pool.
assert(isnumeric(workers)&&~islogical(workers)&&isscalar(workers)&&isfinite(workers)&&workers==fix(workers)&&workers>=1&&workers<=64,'Workers must be an integer from 1 to 64');
pool=[];if workers==1,return;end
assert(exist('parpool','file')==2 && license('test','Distrib_Computing_Toolbox'),'Parallel MATLAB execution requires Parallel Computing Toolbox');
pool=gcp('nocreate');if isempty(pool),pool=parpool('Threads',workers);end
assert(pool.NumWorkers>=workers,'Existing parallel pool has fewer workers than requested');
end
