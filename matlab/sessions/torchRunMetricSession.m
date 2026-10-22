function s = torchRunMetricSession(s,o)
% Advance orthogonal/face-boundary sessions; output only at complete steps.
c=s.config;G=torchSessionGrid(c);count=0;
if ~isempty(o.output_prefix),torchWriteSeries(o.output_prefix,s,o.formats);end
while s.time<c.t_end && count<o.step_limit
    assert(s.steps<c.max_steps,'Maximum steps reached');
    dt=min(torchMetricStep('dt',s,G),c.t_end-s.time);success=false;
    for retry=1:20
        assert(isfinite(dt)&&dt>0&&s.time+dt>s.time,'Integration stalled');old=s.state;lam=s.lam;
        if c.reactive
            [old,lam]=torchSessionReaction(s.state,s.lam,dt/2,c.dimension);
            heated=s;heated.state=old;heated.lam=lam;allowed=torchMetricStep('dt',heated,G);
            if dt>allowed*(1+1e-12),dt=min(dt/2,allowed);continue;end
        end
        stage=s;stage.state=old;stage.lam=lam;
        try
            [hydro,lam,boundary,source]=torchMetricStep('advance',stage,G,dt);
        catch ME
            if strcmp(ME.identifier,'TORCH:MetricPositivity'),dt=dt/2;continue;end
            rethrow(ME);
        end
        U=hydro;
        if c.reactive
            [U,lam]=torchSessionReaction(hydro,lam,dt/2,c.dimension);
            source=source+sum(torchGridOrder(old-s.state+U-hydro,c,'flatten').*G.volume,2);
        end
        s.state=U;s.lam=lam;s.time=min(s.time+dt,c.t_end);s.steps=s.steps+1;
        s.boundary_integral=s.boundary_integral+boundary;s.source_integral=s.source_integral+source;
        success=true;break;
    end
    assert(success,'Could not find positive metric/face step');count=count+1;
    if ~isempty(o.checkpoint)&&o.checkpoint_every>0&&mod(s.steps,o.checkpoint_every)==0,torchWriteCheckpoint(o.checkpoint,s);end
    if ~isempty(o.output_prefix)&&mod(s.steps,o.output_every)==0,torchWriteSeries(o.output_prefix,s,o.formats);end
end
if ~isempty(o.checkpoint),torchWriteCheckpoint(o.checkpoint,s);end
if ~isempty(o.output_prefix),torchWriteSeries(o.output_prefix,s,o.formats);end
end
