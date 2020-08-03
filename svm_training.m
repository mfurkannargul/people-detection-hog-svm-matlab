clc
X = array_img;
% Y = gTruth.LabelData;
Y = gTruth.LabelData
size(X)
size(Y)

Mdl = fitcsvm(X, Y)
% classOrder = SVMModel.ClassNames;
% sv = SVMModel.SupportVectors;