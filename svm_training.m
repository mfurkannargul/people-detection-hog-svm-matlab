clc
X = array_img;
% Y = gTruth.LabelData;
Y = gTruth.LabelDefinitions(1,1);
size(X)
size(Y)

Mdl = fitcsvm(X, Y)
% classOrder = SVMModel.ClassNames;
% sv = SVMModel.SupportVectors;