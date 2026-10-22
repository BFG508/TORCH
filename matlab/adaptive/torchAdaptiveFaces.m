function [L,R,gl,gr,ql,qr] = torchAdaptiveFaces(M,U,Q,c,reconstruct)
[P,g]=torchAdaptivePrimitive(U,Q,c);L=P(:,max(M.left,1));R=P(:,max(M.right,1));gl=g(max(M.left,1));gr=g(max(M.right,1));
ql=Q(:,max(M.left,1));qr=Q(:,max(M.right,1));
for k=1:c.dimension
    for side=[-1,1]
        mask=M.directions==k&M.sides==side;if ~any(mask),continue;end
        face=c.face_boundaries{2*k-(side==-1)};
        if side==-1,[L(:,mask),gl(mask),ql(:,mask)]=torchBoundaryGhost(R(:,mask),gr(mask),qr(:,mask),face,k,side,c.reactive);
        else,[R(:,mask),gr(mask),qr(:,mask)]=torchBoundaryGhost(L(:,mask),gl(mask),ql(:,mask),face,k,side,c.reactive);end
    end
end
if reconstruct&&c.amr.order==2
    gradient=torchAdaptiveGradient(M,P,L,R);
    for side=1:2
        ids=M.left;offset=M.offset_left;target=L;if side==2,ids=M.right;offset=M.offset_right;target=R;end
        mask=ids>0;ix=ids(mask);
        for v=1:size(U,1),change=zeros(sum(mask),1);for a=1:c.dimension,change=change+offset(mask,a).*gradient(ix,a,v);end;target(v,mask)=target(v,mask)+change';end
        if side==1,L=target;else,R=target;end
    end
    for k=1:c.dimension
        for side=[-1,1]
            face=c.face_boundaries{2*k-(side==-1)};if ~strcmp(face.type,'wall'),continue;end
            mask=M.directions==k&M.sides==side;
            if side==-1,L(:,mask)=R(:,mask);L(k+1,mask)=-L(k+1,mask);gl(mask)=gr(mask);
            else,R(:,mask)=L(:,mask);R(k+1,mask)=-R(k+1,mask);gr(mask)=gl(mask);end
        end
    end
end
end
