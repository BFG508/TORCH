function s = torchRunSession(s, varargin)
% Continue to configured end time, or pause at a completed step boundary.
p=inputParser; addParameter(p,'step_limit',Inf); addParameter(p,'checkpoint','');
addParameter(p,'checkpoint_every',0); addParameter(p,'output_prefix','');
addParameter(p,'output_every',1); addParameter(p,'formats',{'vtk','xdmf'}); parse(p,varargin{:});o=p.Results;
assert(o.step_limit>=0 && (isinf(o.step_limit) || o.step_limit==fix(o.step_limit)),'Invalid step limit');
assert(o.checkpoint_every>=0 && o.checkpoint_every==fix(o.checkpoint_every),'Invalid checkpoint interval');
assert(o.output_every>=1 && o.output_every==fix(o.output_every),'Invalid output interval');
c=s.config; dim=c.dimension; h=c.lengths_xyz./c.counts_xyz; volume=prod(h); count=0;
assert(~c.reactive || strcmp(c.kinetic_model,'arrhenius'),'MATLAB restart of ZND is unsupported');
if o.checkpoint_every>0,assert(~isempty(o.checkpoint),'checkpoint_every requires destination');end
if isfield(c,'face_boundaries'),s=torchRunMetricSession(s,o);return;end
if ~isempty(o.output_prefix), torchWriteSeries(o.output_prefix,s,o.formats); end
while s.time<c.t_end && count<o.step_limit
    assert(s.steps<c.max_steps,'Maximum steps reached before requested final time');
    gamma=c.gamma; if c.reactive, gamma=(4.5-s.lam)./(3.5-s.lam); end
    dt=min(sessionDt(s.state,gamma,h,c),c.t_end-s.time);
    assert(isfinite(dt) && dt>0 && s.time+dt>s.time,'Integration stalled');
    start=s.state; old=start; lam=s.lam;
    if c.reactive
        stable=false;
        for retry=1:60
            [old,lam]=torchSessionReaction(start,s.lam,dt/2,dim);
            allowed=sessionDt(old,(4.5-lam)./(3.5-lam),h,c);
            if dt<=allowed*(1+1e-12), stable=true;break;end
            dt=min(dt/2,allowed);
        end
        assert(stable,'Could not find stable reactive step'); gamma=(4.5-lam)./(3.5-lam);
    end
    if dim==1, [U,F]=torchSolver('session_step',old,gamma,h,c,dt);
    elseif dim==2, [U,F]=torchSolver2D('session_step',old,gamma,h,c,dt);
    else, result=torchSolver3D('session_step',old,gamma,h,c,dt);U=result.state;F=result.fluxes;end
    boundary=zeros(dim+2,1);
    for direction=1:dim
        axis=dim+2-direction;flux=F{direction}; lo=repmat({':'},1,dim+1);hi=lo;
        lo{axis}=1;hi{axis}=size(flux,axis);
        boundary=boundary+dt*volume/h(direction)*sum(reshape(flux(hi{:})-flux(lo{:}),dim+2,[]),2);
    end
    source=zeros(dim+2,1);
    if c.reactive
        lam=advect(lam,old,U,F,dt,h,c);
        hydro=U; [U,lam]=torchSessionReaction(U,lam,dt/2,dim);
        source=volume*sum(reshape(old-start+U-hydro,dim+2,[]),2);
    end
    s.state=U;s.lam=lam;s.time=min(s.time+dt,c.t_end);s.steps=s.steps+1;
    s.boundary_integral=s.boundary_integral+boundary;s.source_integral=s.source_integral+source;
    count=count+1;
    if ~isempty(o.checkpoint) && o.checkpoint_every>0 && mod(s.steps,o.checkpoint_every)==0, torchWriteCheckpoint(o.checkpoint,s);end
    if ~isempty(o.output_prefix) && mod(s.steps,o.output_every)==0, torchWriteSeries(o.output_prefix,s,o.formats);end
end
if ~isempty(o.checkpoint), torchWriteCheckpoint(o.checkpoint,s);end
if ~isempty(o.output_prefix), torchWriteSeries(o.output_prefix,s,o.formats);end
end

function dt=sessionDt(U,gamma,h,c)
if c.dimension==1, dt=torchSolver('session_dt',U,gamma,h,c);
elseif c.dimension==2, dt=torchSolver2D('session_dt',U,gamma,h,c);
else, dt=torchSolver3D('session_dt',U,gamma,h,c);end
end


function q=advect(q,old,new,fluxes,dt,h,c)
dim=c.dimension;
if dim==1, rhoq=old(1,:).*q;
else, idx=repmat({':'},1,dim+1);idx{1}=1;rhoq=reshape(old(idx{:}),size(q)).*q;end
for direction=1:dim
    spatial=dim+1-direction;
    if dim==1
        if strcmp(c.boundaries_xyz{1},'Periodic'),pad=[q(end),q,q(1)];else,pad=[q(1),q,q(end)];end
        left=pad(1:end-1);right=pad(2:end);
    else
        n=size(q,spatial);bc=c.boundaries_xyz{direction};index=repmat({':'},1,dim);
        if strcmp(bc,'Periodic'),index{spatial}=[n,1:n,1];else,index{spatial}=[1,1:n,n];end
        pad=q(index{:});l=repmat({':'},1,dim);r=l;l{spatial}=1:n+1;r{spatial}=2:n+2;
        left=pad(l{:});right=pad(r{:});
        if strcmp(bc,'Inflow'),l{spatial}=1;left(l{:})=1;end
    end
    F=fluxes{direction};idx=repmat({':'},1,dim+1);idx{1}=1;
    fshape=size(q);fshape(spatial)=fshape(spatial)+1;if dim==1,fshape=[1,numel(q)+1];end
    mass=reshape(F(idx{:}),fshape);upwind=left;mask=mass<0;upwind(mask)=right(mask);
    rhoq=rhoq-dt/h(direction)*diff(mass.*upwind,1,spatial+(dim==1));
end
idx=repmat({':'},1,dim+1);idx{1}=1;
q=max(0,min(1,rhoq./reshape(new(idx{:}),size(q))));
end
