function M = torchAdaptiveMesh(c,K)
% Dyadic forest with one oriented shared flux for each coarse/fine subface.
d=c.dimension;o=torchAdaptiveOptions(c.amr);counts=c.counts_xyz;maximum=o.max_level;
assert(prod(counts)*2^(d*o.min_level)<=o.max_cells,'AMR capacity cannot cover the minimum level');
if nargin<2
    vectors=cell(1,d);grids=vectors;for k=1:d,vectors{k}=0:counts(k)-1;end
    if d==1,grids{1}=vectors{1};else,[grids{:}]=ndgrid(vectors{:});end
    K=zeros(prod(counts),d+1);for k=1:d,K(:,k+1)=grids{k}(:);end
end
assert(isnumeric(K)&&size(K,2)==d+1&&~isempty(K)&&size(K,1)<=o.max_cells&&all(isfinite(K(:)))&&all(K(:)==fix(K(:))),'Invalid adaptive leaf keys');
K=sortrows(double(K),[1,d+1:-1:2]);levels=K(:,1);N=size(K,1);
assert(all(levels>=0&levels<=maximum)&&all(all(K(:,2:end)>=0&K(:,2:end)<counts.*2.^levels)),'Leaf keys outside configured domain/levels');
total=counts*2^maximum;stride=[1,cumprod(total(1:end-1))];allstride=prod(total);
codes=encode(K);assert(numel(unique(codes))==N,'Duplicate adaptive leaves');
lookup=containers.Map(num2cell(codes'),num2cell(int32(1:N)));
root= floor(K(:,2:end)./2.^levels);rootid=1+root*[1,cumprod(counts(1:end-1))]';
coverage=accumarray(rootid,2.^(-d*levels),[prod(counts),1]);assert(all(coverage==1),'Adaptive leaves leave holes or overlap');
for i=1:N
    parent=K(i,:);
    while parent(1)>0
        parent=[parent(1)-1,floor(parent(2:end)/2)];assert(~isKey(lookup,encode(parent)),'Overlapping adaptive ancestor');
    end
end
integerlo=K(:,2:end).*2.^(maximum-levels);integerhi=integerlo+2.^(maximum-levels);
unit=c.lengths_xyz./total;origin=c.origins_xyz;lo=origin+integerlo.*unit;hi=origin+integerhi.*unit;centers=.5*(lo+hi);widths=hi-lo;
V=torchAdaptiveVolume(c,lo,hi);assert(all(isfinite(V)&V>0),'Invalid adaptive volumes');divergence=zeros(d,N);
for k=1:d,divergence(k,:)=(torchAdaptiveArea(c,k,hi(:,k),lo,hi)-torchAdaptiveArea(c,k,lo(:,k),lo,hi))'./V';end
capacity=2*d*N;L=zeros(capacity,1);R=L;directions=L;sides=L;positions=L;lower=zeros(capacity,d);upper=lower;nf=0;
periodic=false(1,d);for k=1:d,periodic(k)=strcmp(c.face_boundaries{2*k-1}.type,'periodic');end
for i=1:N
    for k=1:d
        a=integerlo(i,:);b=integerhi(i,:);position=b(k);
        if position==total(k)&&~periodic(k),emit(i,0,k,a,b,position,1);else,match(i,k,a,b,position);end
        if a(k)==0&&~periodic(k),emit(0,i,k,a,b,0,-1);end
    end
end
L=L(1:nf);R=R(1:nf);directions=directions(1:nf);sides=sides(1:nf);positions=positions(1:nf);lower=lower(1:nf,:);upper=upper(1:nf,:);
facelo=origin+lower.*unit;facehi=origin+upper.*unit;faceposition=origin(directions)'+positions.*unit(directions)';facecenter=.5*(facelo+facehi);
for k=1:d,mask=directions==k;facecenter(mask,k)=faceposition(mask);end
offL=facecenter-centers(max(L,1),:);offR=facecenter-centers(max(R,1),:);A=zeros(nf,1);
for k=1:d
    mask=directions==k;A(mask)=torchAdaptiveArea(c,k,faceposition(mask),facelo(mask,:),facehi(mask,:));
    mask=directions==k&L>0;offL(mask,k)=.5*widths(L(mask),k);
    mask=directions==k&R>0;offR(mask,k)=-.5*widths(R(mask),k);
end
interior=L>0&R>0;assert(all(abs(levels(L(interior))-levels(R(interior)))<=1),'Adaptive mesh must be face-balanced 2:1');
delta=offL-offR;delta(L==0,:)=-2*offR(L==0,:);delta(R==0,:)=2*offL(R==0,:);weights=1./sum(delta.^2,2);matrix=zeros(N,d,d);
for a=1:d
    for b=1:d
        value=weights.*delta(:,a).*delta(:,b);mask=L>0;matrix(:,a,b)=accumarray(L(mask),value(mask),[N,1]);mask=R>0;matrix(:,a,b)=matrix(:,a,b)+accumarray(R(mask),value(mask),[N,1]);
    end
end
inverse=zeros(size(matrix));
for i=1:N,inverse(i,:,:)=inv(reshape(matrix(i,:,:),d,d));end
M=struct('keys',K,'levels',levels,'low',lo,'high',hi,'centers',centers,'widths',widths,'volumes',V',...
    'divergence',divergence,'left',L,'right',R,'directions',directions,'sides',sides,'areas',A',...
    'offset_left',offL,'offset_right',offR,'displacement',delta,'weights',weights,'inverse',inverse,'interior',interior);

    function code=encode(key)
        code=int64((key(:,1)+1)*allstride+key(:,2:end)*stride');
    end
    function j=findleaf(point)
        for level=0:maximum
            key=[level,floor(point/2^(maximum-level))];code=encode(key);
            if isKey(lookup,code),j=double(lookup(code));return;end
        end
        error('Adaptive mesh has an uncovered neighbor');
    end
    function match(i,k,a,b,position)
        point=.5*(a+b);point(k)=mod(position+.5,total(k));j=findleaf(point);trans=setdiff(1:d,k);
        if all(integerlo(j,trans)<=a(trans))&&all(integerhi(j,trans)>=b(trans)),emit(i,j,k,a,b,position,0);return;end
        middle=floor((a+b)/2);bits=dec2bin(0:2^numel(trans)-1,numel(trans))-'0';
        for child=1:size(bits,1)
            aa=a;bb=b;
            for axis=1:numel(trans),t=trans(axis);if bits(child,axis),aa(t)=middle(t);else,bb(t)=middle(t);end;end
            match(i,k,aa,bb,position);
        end
    end
    function emit(i,j,k,a,b,position,side)
        nf=nf+1;L(nf)=i;R(nf)=j;directions(nf)=k;lower(nf,:)=a;upper(nf,:)=b;positions(nf)=position;sides(nf)=side;
    end
end
