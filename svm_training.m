clc
X = array_img;
Y = 'label'
% SVMModel = fitcsvm(X,Y);
% classOrder = SVMModel.ClassNames
% sv = SVMModel.SupportVectors;
Mdl = fitcsvm(X, Y);
