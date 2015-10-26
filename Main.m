clear; close all; clc

%Load data to be trained
load('AllTrain.csv');
X=AllTrain(:,2:end);
y=AllTrain(:,1);

%Split into training and test sets
train.indices=randperm(size(X, 1),ceil(1*size(X, 1)));
cv.indices=setdiff(1:size(X, 1),train.indices);
Xtrain=X(train.indices,:);
ytrain=y(train.indices,:);
Xcv=X(cv.indices,:);
ycv=y(cv.indices,:);

%Choose network size
input_layer_size=size(X, 2);  
hidden_layer_size=500;   
num_labels=10;   

%Initialize thetas
theta1=randInitializeWeights(input_layer_size,hidden_layer_size);
theta2=randInitializeWeights(hidden_layer_size,num_labels);

%Initialise lambda
lambda=0;

%Create function to return cost and gradients
initJ=0;
gradvec=zeros(numel(theta1)+numel(theta2),1);
thetavec=[theta1(:);theta2(:)];

%Not necessary, but find initial cost and gradients, and verify working
%with numerical gradients
[initJ, gradvec]=nnCostFunction(thetavec,input_layer_size,hidden_layer_size,num_labels,Xtrain,ytrain,lambda);
%numgrad=checkNNGradients(thetavec,input_layer_size,hidden_layer_size,num_labels,Xtrain,ytrain,lambda);

%Set options for fmincg
options=optimset('MaxIter',100);

%Set pointer to cost function
costFunction=@(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

%Start stopwatch
tic               

%Train
[optimtheta, cost, exitflag]=fmincg(costFunction, thetavec, options);

%Stop stopwatch
toc

%Reshape optimal theta to mapping matrices
optimtheta1=reshape(optimtheta(1:hidden_layer_size * (input_layer_size + 1)), ...
               hidden_layer_size, (input_layer_size + 1));

optimtheta2=reshape(optimtheta((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
               num_labels, (hidden_layer_size + 1));

%Predict
pcv=predict(optimtheta1,optimtheta2,Xcv);
ptrain=predict(optimtheta1,optimtheta2,Xtrain);

%Find training and cross validation misclassification rates
mcerrcv=sum(pcv~=ycv)*100/size(ycv,1)
mcerrtrain=sum(ptrain~=ytrain)*100/size(ytrain,1)

%Predict for Kaggle test set
load('AllTest.csv');
Xtest=AllTest;
ptest=predict(optimtheta1,optimtheta2,Xtest);
%Store in format required by Kaggle
completepred=[(1:28000)' ptest];
csvwrite('testpred.csv',completepred);

