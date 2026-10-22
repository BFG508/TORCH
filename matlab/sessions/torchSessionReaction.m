function [out,lam]=torchSessionReaction(U,lam,dt,dim)
flat=reshape(U,dim+2,[]); normal=[flat(1,:);flat(2,:);flat(end,:)-.5*sum(flat(3:end-1,:).^2,1)./flat(1,:)];
[lam,reacted]=torchReaction(normal,lam(:)',dt);
out=U; indices=repmat({':'},1,dim+1);indices{1}=dim+2;
shape=size(U);shape(1)=1;
out(indices{:})=reshape(flat(end,:)+reacted(3,:)-normal(3,:),shape);
spatial=size(U);spatial=spatial(2:end);if dim==1,spatial=[1,spatial];end
lam=reshape(lam,spatial);
end

