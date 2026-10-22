function V = torchAdaptiveVolume(c,lo,hi)
d=c.dimension;g=c.geometry;w=hi-lo;
if strcmp(g,'cartesian'),V=prod(w,2);return;end
s2=(hi(:,1).^2-lo(:,1).^2)/2;
if strcmp(g,'spherical')
    s3=(hi(:,1).^3-lo(:,1).^3)/3;if d==1,V=4*pi*s3;return;end
    V=s3.*(cos(lo(:,2))-cos(hi(:,2)));if d==3,V=V.*w(:,3);else,V=V*2*pi;end
elseif d==1,V=2*pi*s2;
elseif strcmp(g,'cylindrical')&&d==2,V=2*pi*s2.*w(:,2);
else,V=s2.*prod(w(:,2:end),2);end
end
