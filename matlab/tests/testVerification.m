function testVerification
% Independent normal-shock conservation and analytic reference transport.
addpath(fileparts(fileparts(mfilename('fullpath'))));setupTorch;
U=torchReference('shock',8,0,1);left=U(:,1);right=U(:,end);flux=zeros(3,2);
for j=1:2
    Q=left;if j==2,Q=right;end
    u=Q(2)/Q(1);p=.4*(Q(3)-.5*Q(2)^2/Q(1));flux(:,j)=[Q(2);Q(2)*u+p;u*(Q(3)+p)];
end
assert(max(abs(flux(:,1)-flux(:,2)))<1e-13,'Rankine-Hugoniot fluxes do not match');
assert(abs(right(1)-8/3)<1e-14,'Incorrect normal-shock density');
vortex=torchReference('vortex',20,0,2);shifted=torchReference('vortex',20,.5,2);expected=circshift(vortex,[0,1,1]);assert(max(abs(shifted(:)-expected(:)))<4e-15,'Vortex translation');
extruded=torchReference('vortex',20,0,3);axial=reshape(extruded(4,2,:,:),20,20);rho=reshape(vortex(1,:,:),20,20);assert(max(abs(axial(:)-.25*rho(:)))<1e-15,'Vortex axial momentum');
for d=1:3
    period=1/(.2*d+sqrt(1.4*d));a=torchReference('acoustic',12,0,d);b=torchReference('acoustic',12,period,d);assert(max(abs(a(:)-b(:)))<1e-14,'Acoustic phase speed');
end
errors=[];
for n=[24,48,96]
    s=torchVerificationCase('acoustic',n,1,'hllc');s=torchRunSession(s);exact=torchReference('acoustic',n,s.time,1);
    errors(end+1)=mean(abs(s.state(1,:)-exact(1,:))); %#ok<AGROW>
end
assert(all(log2(errors(1:end-1)./errors(2:end))>1.7),'1D acoustic convergence degraded');
fprintf('PASS MATLAB/Octave independent shock, vortex and acoustic references\n');
end
