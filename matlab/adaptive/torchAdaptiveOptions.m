function options = torchAdaptiveOptions(overrides)
% Portable adaptivity settings; no executable callbacks in checkpoints.
options=struct('max_level',2,'min_level',0,'refine_threshold',.08,'coarsen_threshold',.02,'regrid_every',3,'max_cells',100000,'order',2);
if nargin>0
    assert(isstruct(overrides)&&isscalar(overrides),'AMR options must be a scalar structure');keys=fieldnames(overrides);
    for k=1:numel(keys),assert(isfield(options,keys{k}),'Unknown adaptivity option');options.(keys{k})=overrides.(keys{k});end
end
for key={'max_level','min_level','regrid_every','max_cells','order'}
    value=options.(key{1});assert(isnumeric(value)&&isscalar(value)&&isfinite(value)&&value==fix(value),'AMR counts must be integers');
end
assert(0<=options.min_level&&options.min_level<=options.max_level&&options.max_level<=8&&options.regrid_every>=1&&options.max_cells>=2&&options.max_cells<=2000000&&any(options.order==[1,2]),'Invalid AMR level, capacity, interval or order');
low=options.coarsen_threshold;high=options.refine_threshold;assert(isscalar(low)&&isscalar(high)&&all(isfinite([low,high]))&&0<=low&&low<high&&high<=1,'Require 0 <= coarsen threshold < refine <= 1');
end
