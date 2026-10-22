function result = torchSolver3D(varargin)
% Cartesian 3D MUSCL-Hancock/HLLC Euler solver with optional SI Arrhenius chemistry.
% Name/value API: torchSolver3D('preset_name','3d_sod_x','nx',20,...).
% Custom primitive_init is [5,nz,ny,nx]; composition is [nz,ny,nx].
if nargin>0 && any(strcmp(varargin{1},{'session_dt','session_step'}))
    U=varargin{2}; gamma=varargin{3}; h=varargin{4}; c=varargin{5};
    if strcmp(varargin{1},'session_dt'), result=cflDt(U,gamma,h,c.cfl);
    else
        [U,F]=hydro(U,gamma,h,varargin{6},c.limiter,c.boundaries_xyz(end:-1:1),false);
        result=struct('state',U,'fluxes',{F});
    end
    return
end

p=inputParser;
addParameter(p,'preset_name','3d_explosion'); addParameter(p,'primitive_init',[]);
addParameter(p,'nx',20); addParameter(p,'ny',20); addParameter(p,'nz',20);
addParameter(p,'lx',1); addParameter(p,'ly',1); addParameter(p,'lz',1);
addParameter(p,'t_end',[]); addParameter(p,'cfl',0.4); addParameter(p,'gamma',1.4);
addParameter(p,'limiter','VanLeer'); addParameter(p,'bc_x','Open');
addParameter(p,'bc_y','Open'); addParameter(p,'bc_z','Open');
addParameter(p,'is_reactive',false); addParameter(p,'lambda_init',[]);
addParameter(p,'max_steps',100000); parse(p,varargin{:}); o=p.Results;
if isempty(o.t_end)
    if isempty(o.primitive_init) && strcmp(o.preset_name,'3d_ignition'), o.t_end=1e-5; else, o.t_end=.05; end
end
n=[o.nx,o.ny,o.nz]; lengths=[o.lx,o.ly,o.lz];
assert(all(isfinite([n,lengths,o.cfl,o.t_end])) && all(n>=2 & n==fix(n)) && all(lengths>0) && o.cfl>0 && o.cfl<=0.5 && o.t_end>=0,'Invalid 3D grid, CFL or time');
assert(isscalar(o.gamma) && isfinite(o.gamma) && o.gamma>1,'Invalid gamma');
assert(isscalar(o.max_steps) && isfinite(o.max_steps) && o.max_steps>=1 && o.max_steps==fix(o.max_steps),'Invalid maximum steps');
assert(any(strcmp(o.limiter,{'VanLeer','Albada','MINBEE','Superbee','Unos'})),'Unknown limiter');
shape=[o.nz,o.ny,o.nx]; spacing=lengths./n; volume=prod(spacing);
x=((1:o.nx)-0.5)*spacing(1); y=((1:o.ny)-0.5)*spacing(2); z=((1:o.nz)-0.5)*spacing(3);
[Z,Y,X]=ndgrid(z,y,x);
P=o.primitive_init; reactive=o.is_reactive;
if isempty(P)
    P=zeros([5,shape]);
    [zn,yn,xn]=ndgrid(((1:o.nz)-0.5)/o.nz,((1:o.ny)-0.5)/o.ny,((1:o.nx)-0.5)/o.nx);
    switch o.preset_name
        case {'3d_sod_x','3d_sod_y','3d_sod_z'}
            if strcmp(o.preset_name,'3d_sod_x'), coordinate=xn;
            elseif strcmp(o.preset_name,'3d_sod_y'), coordinate=yn;
            else, coordinate=zn; end
            inside=coordinate<=0.5;
            P(1,:,:,:)=reshape(0.125+0.875*inside,[1,shape]);
            P(5,:,:,:)=reshape(0.1+0.9*inside,[1,shape]);
            reactive=false;
        case '3d_explosion'
            inside=(xn-0.5).^2+(yn-0.5).^2+(zn-0.5).^2<=0.2^2;
            P(1,:,:,:)=reshape(0.125+0.875*inside,[1,shape]);
            P(5,:,:,:)=reshape(0.1+0.9*inside,[1,shape]);
            reactive=false;
        case '3d_ignition'
            P(1,:,:,:)=1;
            pressure=101325*ones(shape); pressure(xn<=0.5)=1e6;
            P(5,:,:,:)=reshape(pressure,[1,shape]); reactive=true;
        otherwise, error('Unknown 3D preset');
    end
end
assert(isequal(size(P),[5,shape]) && all(isfinite(P(:))),'Invalid primitive shape or values');
positive=P([1,5],:,:,:); assert(all(positive(:)>0),'Nonpositive initial density/pressure');
boundaries={o.bc_z,o.bc_y,o.bc_x}; pad3(zeros(1,2,2,2),boundaries,false);
lam=[]; gamma=o.gamma;
if reactive
    lam=o.lambda_init; if isempty(lam), lam=ones(shape); end
    assert(isequal(size(lam),shape) && all(isfinite(lam(:))) && all(lam(:)>=0 & lam(:)<=1),'Invalid lambda_init');
    gamma=gammaCalc(lam);
else
    assert(isempty(o.lambda_init),'Composition requires a reactive simulation');
end
P=double(P); U=uprimToUconsv3D(P,gamma); assert(physical(U),'Nonphysical initial state');
initial=sum(reshape(U,5,[]),2)*volume;
boundary=zeros(5,1); source=zeros(5,1); time=0; steps=0;
while time<o.t_end && steps<o.max_steps
    if reactive, gamma=gammaCalc(lam); end
    dt=min(cflDt(U,gamma,spacing,o.cfl),o.t_end-time);
    assert(isfinite(dt) && dt>0 && time+dt>time,'3D integration stalled');
    start=U;
    if reactive
        stable=false;
        for retry=1:60
            [heated,half]=reaction(U,lam,0.5*dt);
            allowed=cflDt(heated,gammaCalc(half),spacing,o.cfl);
            if dt<=allowed*(1+1e-12), stable=true; break; end
            dt=min(0.5*dt,allowed);
        end
        assert(stable,'Could not find stable reactive step');
        old=heated; lam=half; gamma=gammaCalc(lam);
    else, old=U; end
    [U,fluxes]=hydro(old,gamma,spacing,dt,o.limiter,boundaries,false);
    for direction=1:3
        dim=5-direction; F=fluxes{direction}; lo=repmat({':'},1,4); hi=lo;
        lo{dim}=1; hi{dim}=size(F,dim);
        boundary=boundary+dt*volume/spacing(direction)*sum(reshape(F(hi{:})-F(lo{:}),5,[]),2);
    end
    if reactive
        lam=advect(lam,old,U,fluxes,dt,spacing,boundaries);
        hydroEnergy=sum(reshape(U(5,:,:,:),[],1));
        [U,lam]=reaction(U,lam,0.5*dt);
        first=old(5,:,:,:)-start(5,:,:,:);
        source(5)=source(5)+volume*(sum(first(:))+sum(reshape(U(5,:,:,:),[],1))-hydroEnergy);
    end
    time=time+dt; steps=steps+1;
end
assert(time>=o.t_end,'Maximum steps reached before requested final time');
if reactive, gamma=gammaCalc(lam); end
final=sum(reshape(U,5,[]),2)*volume; residual=final-initial+boundary-source;
result=struct('u_final',U,'u_prim_final',uconsvToUprim3D(U,gamma),'lambda_final',lam,...
    'time',time,'steps',steps,'x',x,'y',y,'z',z,'X',X,'Y',Y,'Z',Z,...
    'initial_integral',initial,'final_integral',final,'boundary_integral',boundary,...
    'source_integral',source,'balance_residual',residual,...
    'mass_conservation_error',abs(residual(1))/initial(1),'mass_change_rel',(final(1)-initial(1))/initial(1));
end

function gamma=gammaCalc(lam)
gamma=(4.5-lam)./(3.5-lam);
end

function out=pad3(A,boundaries,vector)
out=A;
for spatial=1:3
    dim=spatial+1; n=size(out,dim); mode=lower(boundaries{spatial});
    switch mode
        case {'periodic','wrap'}, indices=[n-1,n,1:n,1,2];
        case {'open','transmissive','outflow'}, indices=[1,1,1:n,n,n];
        case {'reflective','wall','slip'}, indices=[2,1,1:n,n,n-1];
        otherwise, error('Unknown 3D boundary');
    end
    index=repmat({':'},1,4); index{dim}=indices; out=out(index{:});
    if vector && any(strcmp(mode,{'reflective','wall','slip'}))
        index=repmat({':'},1,4); index{1}=5-spatial; index{dim}=[1,2,n+3,n+4];
        out(index{:})=-out(index{:});
    end
end
end

function F=flux(U,gamma,momentum)
P=uconsvToUprim3D(U,gamma); velocity=P(momentum,:,:,:); pressure=P(5,:,:,:);
F=U.*velocity; F(momentum,:,:,:)=F(momentum,:,:,:)+pressure;
F(5,:,:,:)=velocity.*(U(5,:,:,:)+pressure);
end

function out=active(A)
out=A(:,2:end-1,2:end-1,2:end-1);
end

function delta=slope(A,dim,limiter)
minus=active(A-circshift(A,1,dim)); plus=active(circshift(A,-1,dim)-A);
safe=plus; safe(abs(safe)<1e-14)=1e-14; r=minus./safe; pos=max(r,0);
switch limiter
    case 'VanLeer', phi=2*pos./(1+pos);
    case 'Albada', phi=(pos.^2+pos)./(pos.^2+1);
    case 'MINBEE', phi=max(0,min(r,1));
    case 'Superbee', phi=max(0,max(min(2*r,1),min(r,2)));
    case 'Unos', phi=max(0,min(min(min(2,2*r),(2+r)/3),(1+2*r)/3));
    otherwise, error('Unknown limiter');
end
delta=phi.*plus;
end

function [L,R]=faces(A,dim,isactive)
l=repmat({':'},1,4);
for k=2:4
    if isactive, l{k}=2:size(A,k)-1; else, l{k}=3:size(A,k)-2; end
end
r=l;
if isactive, l{dim}=1:size(A,dim)-1; r{dim}=2:size(A,dim);
else, l{dim}=2:size(A,dim)-2; r{dim}=3:size(A,dim)-1; end
L=A(l{:}); R=A(r{:});
end

function [Unext,fluxes]=hydro(U,gamma,spacing,dt,limiter,boundaries,firstorder)
shape=[size(U,2),size(U,3),size(U,4)];
Up=pad3(U,boundaries,true);
gp=pad3(reshape(gamma.*ones(shape),[1,shape]),boundaries,false);
Ua=active(Up); ga=active(gp); slopes=cell(1,3);
if ~firstorder
    predicted=Ua;
    for direction=1:3
        slopes{direction}=slope(Up,5-direction,limiter);
        d=slopes{direction}; m=direction+1;
        predicted=predicted-0.5*dt/spacing(direction)*(flux(Ua+0.5*d,ga,m)-flux(Ua-0.5*d,ga,m));
    end
    valid=true;
    for direction=1:3
        valid=valid && physical(predicted+0.5*slopes{direction}) && physical(predicted-0.5*slopes{direction});
    end
    if ~valid, [Unext,fluxes]=hydro(U,gamma,spacing,dt,limiter,boundaries,true); return; end
end
fluxes=cell(1,3);
for direction=1:3
    dim=5-direction; m=direction+1;
    if firstorder
        [L,R]=faces(Up,dim,false); [gl,gr]=faces(gp,dim,false);
        pl=uconsvToUprim3D(L,gl); pr=uconsvToUprim3D(R,gr);
        speed=max(abs(pl(m,:,:,:))+sqrt(gl.*pl(5,:,:,:)./pl(1,:,:,:)),abs(pr(m,:,:,:))+sqrt(gr.*pr(5,:,:,:)./pr(1,:,:,:)));
        F=0.5*(flux(L,gl,m)+flux(R,gr,m)-speed.*(R-L));
    else
        [L,~]=faces(predicted+0.5*slopes{direction},dim,true);
        [~,R]=faces(predicted-0.5*slopes{direction},dim,true);
        [gl,gr]=faces(ga,dim,true);
        F=torchHllcGrid(L,R,gl,gr,direction);
    end
    fluxes{direction}=F;
end
Unext=U;
for direction=1:3
    Unext=Unext-dt/spacing(direction)*diff(fluxes{direction},1,5-direction);
end
if ~physical(Unext)
    if ~firstorder, [Unext,fluxes]=hydro(U,gamma,spacing,dt,limiter,boundaries,true); return; end
    error('3D hydrodynamic update produced a nonphysical state');
end
end

function yes=physical(U)
flat=reshape(U,5,[]); internal=flat(5,:)-0.5*sum(flat(2:4,:).^2,1)./flat(1,:);
yes=all(isfinite(flat(:))) && all(flat(1,:)>1e-12) && all(internal>1e-12);
end

function dt=cflDt(U,gamma,spacing,cfl)
P=uconsvToUprim3D(U,gamma); g=reshape(gamma.*ones(size(U,2),size(U,3),size(U,4)),[1,size(U,2),size(U,3),size(U,4)]);
sound=sqrt(g.*P(5,:,:,:)./P(1,:,:,:)); speed=0;
for direction=1:3
    q=abs(P(direction+1,:,:,:))+sound; speed=speed+max(q(:))/spacing(direction);
end
dt=cfl/speed;
end

function [out,next]=reaction(U,lam,dt)
flat=reshape(U,5,[]); normal=[flat(1,:);flat(2,:);flat(5,:)-0.5*sum(flat(3:4,:).^2,1)./flat(1,:)];
[next,reacted]=torchReaction(normal,lam(:)',dt);
out=U; out(5,:,:,:)=reshape(flat(5,:)+reacted(3,:)-normal(3,:),[1,size(lam)]); next=reshape(next,size(lam));
end

function q=advect(q,old,new,fluxes,dt,spacing,boundaries)
shape=size(q); qp=pad3(reshape(q,[1,shape]),boundaries,false); rhoq=old(1,:,:,:).*reshape(q,[1,shape]);
for direction=1:3
    dim=5-direction; [L,R]=faces(qp,dim,false); mass=fluxes{direction}(1,:,:,:);
    upwind=L; mask=mass<0; upwind(mask)=R(mask);
    rhoq=rhoq-dt/spacing(direction)*diff(mass.*upwind,1,dim);
end
q=rhoq./new(1,:,:,:);
assert(all(q(:)>=-1e-10 & q(:)<=1+1e-10),'3D scalar transport left physical interval');
q=reshape(max(0,min(1,q)),shape);
end
