%% TRAIN SVM
clc
%run('extractHOGFeatureVectors.m')
X = array_img;
Y = imds.Labels;
size(X)
size(Y)
Mdl = fitcsvm(X, Y)