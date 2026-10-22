function S = torchAdaptiveSource(M,P,c)
d=c.dimension;g=c.geometry;S=zeros(size(P));if strcmp(g,'cartesian'),return;end
rho=P(1,:);pressure=P(end,:);radial=M.divergence(1,:);S(2,:)=pressure.*radial;
if strcmp(g,'spherical')
    invr=radial/2;S(2,:)=S(2,:)+rho.*sum(P(3:end-1,:).^2,1).*invr;
    if d>=2
        cotr=M.divergence(2,:);S(3,:)=pressure.*cotr-rho.*P(2,:).*P(3,:).*invr;
        if d==3,S(3,:)=S(3,:)+rho.*P(4,:).^2.*cotr;S(4,:)=-rho.*P(4,:).*(P(2,:).*invr+P(3,:).*cotr);end
    end
elseif d>=2&&~(strcmp(g,'cylindrical')&&d==2)
    S(2,:)=S(2,:)+rho.*P(3,:).^2.*radial;S(3,:)=-rho.*P(2,:).*P(3,:).*radial;
end
end
