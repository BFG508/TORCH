function F = torchParallelFlux(kind,L,R,gl,gr,direction,workers,batch_size)
% At most workers independent batches are live; assembly order is fixed.
if nargin<8,batch_size=8192;end
pool=torchWorkers(workers);n=size(L,2);F=zeros(size(L));
if isscalar(gl),gl=repmat(gl,1,n);end;if isscalar(gr),gr=repmat(gr,1,n);end
for start=1:workers*batch_size:n
    count=min(workers,ceil((n-start+1)/batch_size));futures=cell(1,count);ranges=futures;
    try
    for k=1:count
        ix=start+(k-1)*batch_size:min(n,start+k*batch_size-1);ranges{k}=ix;
        if workers==1,F(:,ix)=torchFluxBatch(kind,L(:,ix),R(:,ix),gl(ix),gr(ix),direction);
        else,futures{k}=parfeval(pool,@torchFluxBatch,1,kind,L(:,ix),R(:,ix),gl(ix),gr(ix),direction);end
    end
    if workers>1
        for k=1:count,F(:,ranges{k})=fetchOutputs(futures{k});end
    end
    catch ME
        cancelPending(futures);rethrow(ME);
    end
end
end

function cancelPending(futures)
for k=1:numel(futures)
    f=futures{k};if ~isempty(f),cancel(f);wait(f);end
end
end
