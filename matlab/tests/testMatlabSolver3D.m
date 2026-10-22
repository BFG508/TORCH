function testMatlabSolver3D
% Physical invariants and dimensional reductions; MATLAB and GNU Octave.
addpath(fileparts(fileparts(mfilename('fullpath'))));
setupTorch;
shape=[4,4,4]; P=ones([5,shape]);
P(2,:,:,:)=0.3; P(3,:,:,:)=-0.2; P(4,:,:,:)=0.1;
U=uprimToUconsv3D(P); recovered=uconsvToUprim3D(U);
assert(max(abs(recovered(:)-P(:)))<1e-13,'3D conversion mismatch');
r=torchSolver3D('primitive_init',P,'nx',4,'ny',4,'nz',4,'t_end',.05,'bc_x','Periodic','bc_y','Periodic','bc_z','Periodic');
assert(max(abs(r.u_prim_final(:)-P(:)))<1e-12,'Uniform state not preserved');
assert(max(abs(r.balance_residual))<1e-12,'Five-equation periodic balance');
profiles=cell(1,3); names={'3d_sod_x','3d_sod_y','3d_sod_z'};
for direction=1:3
    n=[4,4,4]; n(direction)=24;
    r=torchSolver3D('preset_name',names{direction},'nx',n(1),'ny',n(2),'nz',n(3),'t_end',.05);
    density=reshape(r.u_prim_final(1,:,:,:),[n(3),n(2),n(1)]);
    if direction==1, profiles{direction}=reshape(density(1,1,:),[],1);
    elseif direction==2, profiles{direction}=reshape(density(1,:,1),[],1);
    else, profiles{direction}=reshape(density(:,1,1),[],1); end
    assert(max(abs(r.balance_residual))<1e-12,'Open boundary balance');
    assert(r.time==.05,'Final time not reached');
end
assert(max(abs(profiles{1}-profiles{2}))<1e-12 && max(abs(profiles{1}-profiles{3}))<1e-12,'Axis permutation mismatch');
limiters={'VanLeer','Albada','MINBEE','Superbee','Unos'};
for k=1:numel(limiters)
    r=torchSolver3D('nx',8,'ny',8,'nz',8,'t_end',.02,'limiter',limiters{k});
    density=reshape(r.u_prim_final(1,:,:,:),[8,8,8]);
    reflected=flip(density,1); rotated=permute(density,[3,2,1]);
    assert(max(abs(density(:)-reflected(:)))<1e-12,'Explosion reflection asymmetry');
    assert(max(abs(density(:)-rotated(:)))<1e-12,'Explosion axis asymmetry');
    assert(all(density(:)>0),'Negative density');
end
% A 2D flow extruded in z must agree over a single common time step.
ny=6; nx=8; nz=3; [y,x]=ndgrid(((1:ny)-.5)/ny,((1:nx)-.5)/nx);
P2=zeros(4,ny,nx); P2(1,:,:)=1+.1*sin(2*pi*x).*cos(2*pi*y);
P2(2,:,:)=.2; P2(3,:,:)=-.1; P2(4,:,:)=1;
U2=P2; U2(2:3,:,:)=P2(2:3,:,:).*P2(1,:,:);
U2(4,:,:)=P2(4,:,:)/.4+.5*P2(1,:,:).*sum(P2(2:3,:,:).^2,1);
P3=zeros(5,nz,ny,nx);
P3([1,2,3,5],:,:,:)=repmat(reshape(P2,[4,1,ny,nx]),[1,nz,1,1]);
[expected,~]=torchSolver2D('custom',U2,1,1,nx,ny,.002,.4,'VanLeer','Periodic','Periodic');
r=torchSolver3D('primitive_init',P3,'nx',nx,'ny',ny,'nz',nz,'t_end',.002,'bc_x','Periodic','bc_y','Periodic','bc_z','Periodic');
actual=reshape(r.u_final([1,2,3,5],1,:,:),[4,ny,nx]);
assert(max(abs(expected(:)-actual(:)))<1e-12,'Extruded 2D mismatch');
% Smooth periodic entropy-wave convergence.
errors=zeros(1,3); sizes=[16,32,64];
for k=1:3
    nx=sizes(k); x=((1:nx)-.5)/nx;
    P=ones(5,4,4,nx); P(1,:,:,:)=repmat(reshape(1+.2*sin(2*pi*x),[1,1,1,nx]),[1,4,4,1]);
    P(2,:,:,:)=1; P(3:4,:,:,:)=0;
    r=torchSolver3D('primitive_init',P,'nx',nx,'ny',4,'nz',4,'t_end',.1,'bc_x','Periodic','bc_y','Periodic','bc_z','Periodic');
    rho=reshape(r.u_prim_final(1,1,1,:),1,[]);
    errors(k)=mean(abs(rho-(1+.2*sin(2*pi*(x-.1)))));
end
assert(errors(1)/errors(2)>3 && errors(2)/errors(3)>3,'Smooth mesh convergence below expected order');
% Species must move in every direction without reacting at cold SI conditions.
P=ones(5,8,8,8); P(2,:,:,:)=1; P(3,:,:,:)=.5; P(4,:,:,:)=.3; P(5,:,:,:)=100;
lam=zeros(8,8,8); lam(3:4,3:4,3:4)=1;
r=torchSolver3D('primitive_init',P,'nx',8,'ny',8,'nz',8,'t_end',.02,'is_reactive',true,'lambda_init',lam,'bc_x','Periodic','bc_y','Periodic','bc_z','Periodic');
species=reshape(r.u_final(1,:,:,:),size(lam)).*r.lambda_final;
assert(abs(sum(species(:))-sum(lam(:)))<1e-10,'Species not conserved');
for coord={r.X,r.Y,r.Z}
    q=coord{1}; assert(sum(species(:).*q(:))>sum(lam(:).*q(:)),'Species did not advect');
end
r=torchSolver3D('preset_name','3d_ignition','nx',4,'ny',4,'nz',4,'t_end',1e-5);
assert(min(r.lambda_final(:))<.99,'Hot mixture did not react');
assert(abs(r.balance_residual(5))/r.final_integral(5)<1e-12,'Reactive energy balance');
for bc={'Reflective','Open'}
    P=ones(5,4,4,4); P(2,:,:,:)=1; P(3,:,:,:)=.5; P(4,:,:,:)=-.3;
    r=torchSolver3D('primitive_init',P,'nx',4,'ny',4,'nz',4,'t_end',.04,'bc_x',bc{1},'bc_y',bc{1},'bc_z',bc{1});
    assert(max(abs(r.balance_residual))<1e-12,'Boundary-integrated balance');
    if strcmp(bc{1},'Reflective'), assert(max(abs(r.boundary_integral([1,5])))<1e-12,'Wall leaks mass/energy'); end
end
rejected=false;
try, torchSolver3D('nx',4,'ny',4,'nz',4,'t_end',1,'max_steps',1); catch, rejected=true; end
assert(rejected,'Premature integration incorrectly returned success');
fprintf('ALL 3D MATLAB/OCTAVE PHYSICAL AND CONVERGENCE CHECKS PASSED!\n');
end
