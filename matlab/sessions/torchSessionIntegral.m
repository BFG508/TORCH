function value = torchSessionIntegral(s)
% Physical volume integral in the local momentum basis.
c=s.config;
if ~isfield(c,'geometry') || strcmp(c.geometry,'cartesian')
    value=sum(reshape(s.state,c.dimension+2,[]),2)*prod(c.lengths_xyz./c.counts_xyz);
else
    G=torchMetricGrid(c);value=sum(torchGridOrder(s.state,c,'flatten').*G.volume,2);
end
end
