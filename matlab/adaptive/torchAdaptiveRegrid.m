function s = torchAdaptiveRegrid(s,refine,coarsen)
% Conservative sibling restriction/prolongation and deterministic 2:1 closure.
if nargin<3,coarsen=true;end
M=s.mesh;c=s.config;o=c.amr;U=s.state;N=size(U,2);d=c.dimension;Q=zeros(0,N);if c.reactive,Q=s.lam;end
[L,R,gl,gr]=torchAdaptiveFaces(M,U,Q,c,false);jumps=max(abs(L([1,end],:)-R([1,end],:))./max(abs(L([1,end],:))+abs(R([1,end],:)),1e-14),[],1)';indicator=zeros(N,1);
for side=1:2,ids=M.left;if side==2,ids=M.right;end;mask=ids>0;indicator=max(indicator,accumarray(ids(mask),jumps(mask),[N,1],@max,0));end
if nargin<2,refine=indicator>o.refine_threshold;end
assert(numel(refine)==N,'Refinement mask must match leaves');tag=(logical(refine(:))|M.levels<o.min_level)&M.levels<o.max_level;
i=M.left(M.interior);j=M.right(M.interior);
for iteration=1:o.max_level+1
    old=tag;a=M.levels(i)+tag(i);b=M.levels(j)+tag(j);tag(j(a>b+1))=true;tag(i(b>a+1))=true;if isequal(old,tag),break;end
end
parents=zeros(0,d+1);groups={};eligible=find(M.levels>o.min_level&~tag&indicator<o.coarsen_threshold);
if coarsen&&~isempty(eligible)
    candidates=[M.levels(eligible)-1,floor(M.keys(eligible,2:end)/2)];[uniqueparents,~,membership]=unique(candidates,'rows');
    for g=1:size(uniqueparents,1)
        ids=eligible(membership==g);if numel(ids)~=2^d,continue;end
        inside=false(N,1);inside(ids)=true;neighbor=[j(inside(i)&~inside(j));i(inside(j)&~inside(i))];
        if any(M.levels(neighbor)+tag(neighbor)>uniqueparents(g,1)+1),continue;end
        parents(end+1,:)=uniqueparents(g,:);groups{end+1}=ids; %#ok<AGROW>
    end
end
count=N+sum(tag)*(2^d-1)-numel(groups)*(2^d-1);budget=count>o.max_cells;if budget,tag(:)=false;end
if ~any(tag)&&isempty(groups),s.regrids=s.regrids+1;s.budget_limited=budget;return;end
count=N+sum(tag)*(2^d-1)-numel(groups)*(2^d-1);keys=zeros(count,d+1);states=zeros(size(U,1),count);fractions=zeros(size(Q,1),count);removed=false(N,1);next=0;
for g=1:numel(groups)
    ids=groups{g};removed(ids)=true;weights=M.volumes(ids);next=next+1;keys(next,:)=parents(g,:);states(:,next)=sum(U(:,ids).*weights,2)/sum(weights);
    if ~isempty(Q),fractions(:,next)=max(0,min(1,sum(U(1,ids).*Q(:,ids).*weights,2)/sum(U(1,ids).*weights)));end
end
gradient=[];if any(tag)&&o.order==2,gradient=torchAdaptiveGradient(M,U,torchAdaptiveConservative(L,gl),torchAdaptiveConservative(R,gr));end
for cell=1:N
    if removed(cell),continue;end
    if ~tag(cell),next=next+1;keys(next,:)=M.keys(cell,:);states(:,next)=U(:,cell);fractions(:,next)=Q(:,cell);continue;end
    children=torchAdaptiveChildren(M.keys(cell,:));nc=size(children,1);lo=c.origins_xyz+children(:,2:end).*c.lengths_xyz./c.counts_xyz/2^children(1,1);hi=lo+c.lengths_xyz./c.counts_xyz/2^children(1,1);weights=torchAdaptiveVolume(c,lo,hi);
    values=repmat(U(:,cell),1,nc);
    if ~isempty(gradient)
        change=(.5*(lo+hi)-M.centers(cell,:))*reshape(gradient(cell,:,:),d,size(U,1));change=change-sum(change.*weights,1)/sum(weights);trial=values+change';if torchAdaptivePhysical(trial),values=trial;end
    end
    ids=next+(1:nc);keys(ids,:)=children;states(:,ids)=values;fractions(:,ids)=repmat(Q(:,cell),1,nc);next=next+nc;
end
[keys,order]=sortrows(keys,[1,d+1:-1:2]);candidate=torchAdaptiveMesh(c,keys);
s.mesh=candidate;s.state=states(:,order);s.lam=[];if c.reactive,s.lam=fractions(:,order);end
s.regrids=s.regrids+1;s.mesh_revision=s.mesh_revision+1;s.budget_limited=budget;
end
