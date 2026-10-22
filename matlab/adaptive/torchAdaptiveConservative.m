function U = torchAdaptiveConservative(P,g)
U=P;U(2:end-1,:)=P(2:end-1,:).*P(1,:);U(end,:)=P(end,:)./(g-1)+.5*P(1,:).*sum(P(2:end-1,:).^2,1);
end
