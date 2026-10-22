function keys = torchAdaptiveChildren(parent)
d=numel(parent)-1;bits=dec2bin(0:2^d-1,d)-'0';bits=bits(:,end:-1:1);
keys=[repmat(parent(1)+1,2^d,1),2*parent(2:end)+bits];
end
