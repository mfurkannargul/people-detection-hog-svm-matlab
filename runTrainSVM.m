%% TRAIN SVM
clc
run('resizeImages.m')
run('extractHogVector.m')
load('gTruth.mat')
X = array_img;
Y = gTruth.LabelData
size(X)
size(Y)
%Mdl = fitcsvm(X, Y);
