setwd("C:/Users/Sagar/OneDrive/Documents/Learning/Handwritten")
if(!require(plyr)) install.packages("plyr")
library(plyr)
if(!require(e1071)) install.packages("e1071")
library(e1071)
if(!require(caret)) install.packages("caret")
library(caret)
if(!require(doParallel)) install.packages("doParallel")
library(doParallel)
if(!require(pROC)) install.packages("pROC")
library(pROC)
if(!require(kernlab)) install.packages("kernlab")
library(kernlab)

#load data
x<-read.csv("AllTrain.csv", header=FALSE)
y<-x[,1]
y<-make.names(as.factor(y))
y<-as.factor(y)
x<-x[,-c(1)]

#remove near-zero-variance variables
nzv<-nearZeroVar(x)
filteredDescr <- x[, -nzv]
dim(filteredDescr)
x<-filteredDescr

#define trainControl
fitControl<-trainControl(method="cv",
                         number=4,
                         classProbs=FALSE,
                         seeds=NULL,
                         allowParallel=TRUE)

#define tuning grid
svmgrid<-expand.grid(.degree=c(3,4,5),
                     .scale=c(0.01,0.02,0.05,0.1,0.2,0.5),
                     .C=1)


#register parallel backend
cl <- makeCluster(detectCores())
registerDoParallel(cl)
getDoParWorkers()

#Check training start time
tic<-Sys.time()

#train model
svmfit<-train(x,
              y,
              method="svmPoly",
              trControl=fitControl,
              tuneGrid=svmgrid,
              metric="Accuracy")

#Check training end time
toc<-Sys.time()

#print training time
toc-tic

#view parameter results
svmfit

#view best parameters
svmfit$bestTune

#view final model used
svmfit$finalModel

#print confusion matrix
pred<-predict(svmfit, newdata=xtrain)
table(pred,y)

#print feature importance in predicting classes
varImp(svmfit, scale=TRUE)

#stop cluster
stopCluster(cl)

#load Kaggle test data and predict
testdata<-read.csv("AllTest.csv", header=FALSE)
test <- testdata[, -nzv]
pred<-predict(svmfit, newdata=test)

#store in format required by Kaggle
preddf<-data.frame(pred)
levels(preddf$pred) <- c(levels(preddf$pred), c("0","1","2","3","4","5","6","7","8","9"))
preddf$pred[preddf$pred=="X0"]<-"0"
preddf$pred[preddf$pred=="X1"]<-"1"
preddf$pred[preddf$pred=="X2"]<-"2"
preddf$pred[preddf$pred=="X3"]<-"3"
preddf$pred[preddf$pred=="X4"]<-"4"
preddf$pred[preddf$pred=="X5"]<-"5"
preddf$pred[preddf$pred=="X6"]<-"6"
preddf$pred[preddf$pred=="X7"]<-"7"
preddf$pred[preddf$pred=="X8"]<-"8"
preddf$pred[preddf$pred=="X9"]<-"9"
preddf$ImageId=1:nrow(preddf)
preddf<-preddf[,c(2,1)]
names(preddf)[2]<-"Label"
write.csv(preddf,"preddf.csv",row.names=FALSE)

#THIS TUNED MDOEL PRODUCES AN ACCURACY OF AROUND 0.985, WHICH IS AROUND RANK 100 ON KAGGLE
#THIS CAN BE FURTHER IMPROVED, PERHAPS BY TUNING SCALE AND PERHAPS DEGREE TOO (TO BE DONE LATER)