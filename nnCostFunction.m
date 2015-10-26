function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%Function to calculate cost and cost gradients

%Reshape thetas into mapping matrices
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         
%Initialize cost
J = 0;

%Reconstruct y to a multiclass structure (for output layer)
ynew=zeros(length(y),num_labels);
for k=1:num_labels,
  ynew(:,k)=(y==mod(k,10));
end;
y=ynew;  

%Implement forward propagation
a1=X;
a1wb=[ones(length(X),1),X];
z2=a1wb*Theta1'; 
a2=sigmoid(z2);
a2wb=[ones(length(a2),1),a2];
z3=a2wb*Theta2';
a3=sigmoid(z3);
htheta=a3;

%Calculate cost 
J=(-1/m)*sum(sum(y.*log(htheta)+(1-y).*log(1-htheta)));
%Add regularization parameter (L2)
J=J+(lambda/(2*m))*(sum(sum(Theta1(:,2:size(Theta1,2)).^2))+sum(sum(Theta2(:,2:size(Theta2,2)).^2)));

%Create matrices to hold gradients
capdelta1=zeros(size(Theta1));
capdelta2=zeros(size(Theta2));
grad1=zeros(size(Theta1));
grad2=zeros(size(Theta2));

%Calculate for each instance and sum over
for i=1:m,
  %Forward propagation
  a1s=(X(i,:))';
  a1wbs=[1;a1s];
  z2s=Theta1*a1wbs;
  %Sigmoid activation on hidden layer
  a2s=sigmoid(z2s);
  a2wbs=[1;a2s];
  z3s=Theta2*a2wbs;
  a3s=sigmoid(z3s);
  hthetas=a3s;
  %Backpropagation
  delta3=hthetas-y(i,:)';
  delta2wb=(Theta2'*delta3).*sigmoidGradient([1;z2s]);
  delta2=delta2wb(2:end);
  capdelta1=capdelta1+delta2*a1wbs';
  capdelta2=capdelta2+delta3*a2wbs';
end;
%Calculate gradients
grad1=(1/m)*capdelta1;
grad2=(1/m)*capdelta2;
grad1(:,2:size(grad1,2))=grad1(:,2:size(grad1,2))+(lambda/m)*Theta1(:,2:size(Theta1,2));
grad2(:,2:size(grad2,2))=grad2(:,2:size(grad2,2))+(lambda/m)*Theta2(:,2:size(Theta2,2));

%Reshape to 1D vector
grad=[grad1(:);grad2(:)];






















% -------------------------------------------------------------


end
