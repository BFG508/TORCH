function A = torchAdaptiveArea(c,k,position,lo,hi)
d=c.dimension;g=c.geometry;w=hi-lo;
if strcmp(g,'cartesian'),A=prod(w(:,setdiff(1:d,k)),2);return;end
if d==1,if strcmp(g,'spherical'),A=4*pi*position.^2;else,A=2*pi*position;end;return;end
s2=(hi(:,1).^2-lo(:,1).^2)/2;
if strcmp(g,'spherical')
    az=2*pi;if d==3,az=w(:,3);end
    if k==1,A=position.^2.*(cos(lo(:,2))-cos(hi(:,2))).*az;
    elseif k==2,A=s2.*sin(position).*az;
    else,A=s2.*w(:,2);end
elseif strcmp(g,'cylindrical')&&d==2
    if k==1,A=2*pi*position.*w(:,2);else,A=2*pi*s2;end
elseif k==1,A=position.*prod(w(:,2:end),2);
elseif k==2,A=w(:,1);if d==3,A=A.*w(:,3);end
else,A=s2.*w(:,2);end
end
