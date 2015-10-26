function p = predict(Theta1, Theta2, X)

p = zeros(size(X, 1), 1);

a1=X;
a1wb=[ones(size(X, 1),1),X];
z2=a1wb*Theta1'; 
a2=sigmoid(z2);
a2wb=[ones(size(a2, 1),1),a2];
z3=a2wb*Theta2';
a3=sigmoid(z3);
htheta=a3;

[dummy, indices] = max(htheta, [], 2);
p=mod(indices,10);




end
