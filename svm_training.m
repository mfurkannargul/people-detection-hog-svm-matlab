clc
X = array_img;
% SVMModel = fitcsvm(X,Y);
% classOrder = SVMModel.ClassNames
% sv = SVMModel.SupportVectors;
classifier = fitcecoc(array_img, 'people');
