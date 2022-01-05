close all
clear all
clc

data = csvread('heart.csv');
%disp(data);
X = data(:,1:13);
Y = data(:,14);
classNames = {'0', '1'}


% Cross varidation (train: 70%, test: 30%)
%cv = cvpartition(size(data,1),'HoldOut',0.3);
%idx = cv.test;

% Separate to training and test data
%dataTrain = data(~idx,:);
%dataTest  = data(idx,:);

%X_train = dataTrain(:,1:13);
%Y_train = dataTrain(:,14);

%X_test = dataTest(:,1:13);
%Y_test = dataTest(:,14);



model = fitcnb(X,Y,'Distribution', 'normal','CrossVal','on')
%model = fitcnb(X,Y,'CrossVal','on')
%predict =  kfoldPredict(model)
%disp(predict)

%make predictions and calculate the score for ROC curve
[predict, classifScore] =  kfoldPredict(model)

%plot confusion matrix
confusionchart(Y, predict)
C = confusionmat(Y, predict)
%disp(C)

%extract the values from confusion matrix
truePositive = C(2,2)
falsePositive = C(2,1)
falseNegative = C(1,2)
trueNegative = C(1,1)

%calculate the measures used to evaluate the model
accuracy = (truePositive + trueNegative)/(truePositive + trueNegative + falsePositive + falseNegative)
precision = truePositive/(truePositive + falsePositive)
recall = truePositive/(truePositive + falseNegative)
fScore = 2*truePositive/(2*truePositive + falsePositive + falseNegative)

%calculate and plot the the misclassification rate and generalization error
kflc = kfoldLoss(model, 'Mode', 'average');
figure;
plot(kflc);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');

estGenError = kflc(end)

%plot roc curve and calculate auc
[xx,yy,~,auc] = perfcurve(Y, classifScore(:,2),'1');
figure;
plot(xx,yy)
xlabel('False positive rate');
ylabel('True positive rate');
disp(auc)

%tune the hyperparameters
%rng default
%Mdl = fitcnb(X,Y,'ClassNames',classNames,'OptimizeHyperparameters','auto',...
   % 'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
   % 'expected-improvement-plus'))


