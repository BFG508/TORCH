function s = torchRunAdaptiveSession(s,o)
c=s.config;count=0;if ~isempty(o.output_prefix),torchWriteSeries(o.output_prefix,s,o.formats);end
while s.time<c.t_end&&count<o.step_limit
    assert(s.steps<c.max_steps,'Maximum steps reached');dt=min(torchAdaptiveOperator('dt',s),c.t_end-s.time);success=false;
    for retry=1:24
        assert(isfinite(dt)&&dt>0&&s.time+dt>s.time,'Adaptive integration stalled');old=s.state;lam=s.lam;
        if c.reactive
            [old,lam]=reaction(s.state,s.lam,dt/2);heated=s;heated.state=old;heated.lam=lam;allowed=torchAdaptiveOperator('dt',heated);
            if dt>allowed*(1+1e-12),dt=min(dt/2,allowed);continue;end
        end
        stage=s;stage.state=old;stage.lam=lam;
        try,[hydro,lam,boundary,source]=torchAdaptiveOperator('advance',stage,dt);
        catch ME,if strcmp(ME.identifier,'TORCH:AdaptivePositivity'),dt=dt/2;continue;end;rethrow(ME);end
        U=hydro;if c.reactive,[U,lam]=reaction(hydro,lam,dt/2);source=source+sum((old-s.state+U-hydro).*s.mesh.volumes,2);end
        assert(torchAdaptivePhysical(U),'Nonphysical adaptive reaction');candidate=s;candidate.state=U;candidate.lam=[];if c.reactive,candidate.lam=lam;end
        candidate.time=min(s.time+dt,c.t_end);candidate.steps=s.steps+1;
        candidate.boundary_integral=s.boundary_integral+boundary;candidate.source_integral=s.source_integral+source;
        if mod(candidate.steps,c.amr.regrid_every)==0,candidate=torchAdaptiveRegrid(candidate);end
        s=candidate;success=true;break;
    end
    assert(success,'Could not find a positive adaptive step');count=count+1;
    if ~isempty(o.checkpoint)&&o.checkpoint_every>0&&mod(s.steps,o.checkpoint_every)==0,torchWriteCheckpoint(o.checkpoint,s);end
    if ~isempty(o.output_prefix)&&mod(s.steps,o.output_every)==0,torchWriteSeries(o.output_prefix,s,o.formats);end
end
if ~isempty(o.checkpoint),torchWriteCheckpoint(o.checkpoint,s);end
if ~isempty(o.output_prefix),torchWriteSeries(o.output_prefix,s,o.formats);end
end

function [out,lam]=reaction(U,lam,dt)
normal=[U(1,:);U(2,:);U(end,:)-.5*sum(U(3:end-1,:).^2,1)./U(1,:)];[lam,reacted]=torchReaction(normal,lam,dt);out=U;out(end,:)=U(end,:)+reacted(end,:)-normal(end,:);
end
