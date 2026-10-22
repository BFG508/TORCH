function gradient = torchAdaptiveGradient(M,values,left,right)
% Weighted least squares with a cellwise neighbor-extrema limiter.
N=size(values,2);nv=size(values,1);d=size(M.centers,2);nf=numel(M.left);difference=(right-left)';
vector=zeros(N,d,nv);low=values';high=low;
for side=1:2
    ids=M.left;neighbor=right;if side==2,ids=M.right;neighbor=left;end
    mask=ids>0;ix=ids(mask);
    for v=1:nv
        low(:,v)=min(low(:,v),accumarray(ix,neighbor(v,mask)',[N,1],@min,Inf));
        high(:,v)=max(high(:,v),accumarray(ix,neighbor(v,mask)',[N,1],@max,-Inf));
        for a=1:d
            contribution=M.weights.*M.displacement(:,a).*difference(:,v);
            vector(:,a,v)=vector(:,a,v)+accumarray(ix,contribution(mask),[N,1]);
        end
    end
end
gradient=zeros(size(vector));
for a=1:d,for b=1:d,gradient(:,a,:)=gradient(:,a,:)+M.inverse(:,a,b).*vector(:,b,:);end;end
theta=ones(N,nv);
for side=1:2
    ids=M.left;offset=M.offset_left;if side==2,ids=M.right;offset=M.offset_right;end
    mask=ids>0;ix=ids(mask);
    for v=1:nv
        change=zeros(sum(mask),1);for a=1:d,change=change+offset(mask,a).*gradient(ix,a,v);end
        allowed=high(ix,v)-values(v,ix)';negative=change<0;allowed(negative)=low(ix(negative),v)-values(v,ix(negative))';
        ratio=ones(size(change));active=abs(change)>1e-14;ratio(active)=allowed(active)./change(active);
        theta(:,v)=min(theta(:,v),accumarray(ix,max(0,min(1,ratio)),[N,1],@min,1));
    end
end
for v=1:nv,gradient(:,:,v)=gradient(:,:,v).*theta(:,v);end
end
