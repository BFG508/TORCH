function [P,g,Q] = torchBoundaryGhost(P,g,Q,f,d,side,reactive)
% Local normal characteristic conditions; side is outward normal sign.
normal=d+1;kind=f.type;
if strcmp(kind,'wall'),P(normal,:)=-P(normal,:);
elseif any(strcmp(kind,{'inflow','stagnation_inlet'}))
    interiorGamma=g;fraction=1;if isfield(f,'lambda'),fraction=f.lambda;end
    if reactive,g(:)=(4.5-fraction)/(3.5-fraction);end
    for k=1:size(Q,1),names={'lambda','beta'};value=1;if isfield(f,names{k}),value=f.(names{k});end;Q(k,:)=value;end
    if strcmp(kind,'inflow'),P=repmat(f.primitive(:),1,size(P,2));
    else
        a=sqrt(interiorGamma.*P(end,:)./P(1,:));J=-side*P(normal,:)-2*a./(interiorGamma-1);a0sq=g*f.gas_constant*f.total_temperature;
        A=(g+1)./(g-1);D=J.^2-A.*(.5*(g-1).*J.^2-a0sq);assert(all(D>=0),'No reservoir-inlet solution');
        a=(-J+sqrt(D))./A;v=J+2*a./(g-1);choked=v>=a;
        a(choked)=sqrt(2*a0sq(choked)./(g(choked)+1));v(choked)=a(choked);
        assert(all(v>=0 & a>0),'Reservoir cannot supply inflow');P(2:end-1,:)=0;P(normal,:)=-side*v;
        P(end,:)=f.total_pressure*(a.^2./a0sq).^(g./(g-1));P(1,:)=g.*P(end,:)./a.^2;
    end
elseif strcmp(kind,'pressure_outlet')
    a=sqrt(g.*P(end,:)./P(1,:));un=side*P(normal,:);sub=un<a;
    rho=P(1,:).*(f.pressure./P(end,:)).^(1./g);anew=sqrt(g*f.pressure./rho);v=un+2*(a-anew)./(g-1);back=sub & (v<0 | un<0);
    assert(~any(back)||isfield(f,'backflow'),'Pressure outlet backflow requires explicit primitives');
    P(1,sub)=rho(sub);P(end,sub)=f.pressure;P(normal,sub)=side*v(sub);
    if any(back)
        P(:,back)=repmat(f.backflow(:),1,sum(back));fraction=1;if isfield(f,'lambda'),fraction=f.lambda;end
        if reactive,g(back)=(4.5-fraction)/(3.5-fraction);end
        for k=1:size(Q,1),names={'lambda','beta'};value=1;if isfield(f,names{k}),value=f.(names{k});end;Q(k,back)=value;end
    end
end
end
