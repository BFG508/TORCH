function F=torchParallelLegacy1D(kernel,U,W,g,variant,workers)
% Retain the legacy one-dimensional wave estimates and arithmetic exactly.
pool=torchWorkers(workers);n=size(W,2);F=zeros(3,n);batch=8192;
for start=1:workers*batch:n
    count=min(workers,ceil((n-start+1)/batch));futures=cell(1,count);ranges=futures;
    try
        for k=1:count
            first=start+(k-1)*batch;last=min(n,start+k*batch-1);ix=first:last;ranges{k}=ix;
            gas=g;if ~isscalar(g),gas=g(first:last+1);end
            futures{k}=parfeval(pool,kernel,1,U(:,2*first-1:2*last),W(:,ix),gas,variant,1);
        end
        for k=1:count,F(:,ranges{k})=fetchOutputs(futures{k});end
    catch ME
        for k=1:count,if ~isempty(futures{k}),cancel(futures{k});wait(futures{k});end;end
        rethrow(ME);
    end
end
end
