function testGeometry
% Independent physical volumes, rest balance, expansion and inlet invariants.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
for dim=1:3
    for kind={'cartesian','cylindrical','annular','spherical'}
        g=kind{1};origin=.3*ones(1,dim);L=.5*ones(1,dim);
        if dim<3 && ~strcmp(g,'annular'),origin(:)=0;end
        if strcmp(g,'spherical') && dim==2,L(2)=pi;end
        P=ones([dim+2,repmat(6,1,dim)]);idx=repmat({':'},1,dim+1);idx{1}=2:dim+1;P(idx{:})=0;
        s=torchCreateSession(P,L,'origins',origin,'geometry',g,'t_end',.02);G=torchMetricGrid(s.config);
        r0=origin(1);r1=r0+L(1);
        if strcmp(g,'cartesian'),expected=prod(L);
        elseif strcmp(g,'spherical')
            solid=4*pi;if dim>=2,solid=(cos(origin(2))-cos(origin(2)+L(2)))*2*pi;end
            if dim==3,solid=(cos(origin(2))-cos(origin(2)+L(2)))*L(3);end
            expected=(r1^3-r0^3)/3*solid;
        else
            factor=2*pi;if dim==2,if strcmp(g,'cylindrical'),factor=2*pi*L(2);else,factor=L(2);end;elseif dim==3,factor=L(2)*L(3);end
            expected=(r1^2-r0^2)/2*factor;
        end
        assert(abs(sum(G.volume)-expected)<1e-12,'Incorrect physical volume');
        final=torchRunSession(s);assert(max(abs(final.state(:)-s.state(:)))<3e-14,'Uniform rest imbalance');
    end
end
for kind={'cylindrical','spherical'}
    D=2;if strcmp(kind{1},'spherical'),D=3;end;errors=[];
    for n=[24,48,96]
        r=.5+((1:n)-.5)/n;P=[ones(1,n);.2*r;ones(1,n)];
        s=torchCreateSession(P,1,'origins',.5,'geometry',kind{1},'t_end',.02);s=torchRunSession(s);
        f=1+.2*s.time;rho=s.state(1,:);velocity=s.state(2,:)./rho;pressure=.4*(s.state(3,:)-.5*rho.*velocity.^2);
        exact=[ones(1,n)*f^(-D);.2*r/f;ones(1,n)*f^(-1.4*D)];mask=r>.7 & r<1.3;
        difference=abs([rho;velocity;pressure]-exact);errors(end+1)=mean(mean(difference(:,mask))); %#ok<AGROW>
    end
    assert(all(errors(1:end-1)./errors(2:end)>2.5),'Radial expansion did not converge');
end
P=[1.2;20;101325];f=struct('type','stagnation_inlet','total_pressure',110000,'total_temperature',300,'gas_constant',287);
[inlet,g,~]=torchBoundaryGhost(P,1.4,zeros(0,1),f,1,-1,false);a=sqrt(g*inlet(end)/inlet(1));M=inlet(2)/a;
assert(abs(inlet(end)*(1+.5*(g-1)*M^2)^(g/(g-1))-110000)<1e-8,'Incorrect total pressure');
assert(abs(inlet(end)/(287*inlet(1))+inlet(2)^2/(2*1.4*287/.4)-300)<1e-10,'Incorrect total temperature');
f=struct('type','pressure_outlet','pressure',90000);[outlet,~,~]=torchBoundaryGhost(P,1.4,zeros(0,1),f,1,1,false);
assert(abs(outlet(end)/outlet(1)^1.4-P(end)/P(1)^1.4)<1e-8,'Outlet entropy invariant');
fprintf('PASS MATLAB/Octave metric volumes, rest balance, analytic expansion and characteristic invariants\n');
end
