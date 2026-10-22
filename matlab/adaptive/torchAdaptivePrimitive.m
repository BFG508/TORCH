function [P,g] = torchAdaptivePrimitive(U,Q,c)
g=c.gamma*ones(1,size(U,2));if c.reactive,g=(4.5-Q(1,:))./(3.5-Q(1,:));end
P=U;P(2:end-1,:)=U(2:end-1,:)./U(1,:);P(end,:)=(g-1).*(U(end,:)-.5*sum(U(2:end-1,:).^2,1)./U(1,:));
end
