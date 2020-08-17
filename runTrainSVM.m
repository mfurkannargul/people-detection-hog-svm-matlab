%% TRAIN SVM
clc
run('resizeImages.m')
run('extractHogVector.m')
% load('gTruth.mat')
X = array_img;
Y = imds.Labels;
size(X)
size(Y)
Mdl = fitcsvm(X, Y)