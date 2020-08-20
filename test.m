clc
%% LOAD SVM CLASSIFIER TRAINED
load SVMModel

%% LOAD TEST IMAGES
folderTest = 'C:\Users\furka\Documents\GitHub\people-detection-hog-svm-matlab\test_images';
imdsTest = imageDatastore(folderTest, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% DISPLAY A SAMPLE TEST IMAGE
t = 4;
imageTest = readimage(imdsTest, t);
figure(3);
imshow(imageTest);
title('A Sample Test Image');

%% MAKE PREDICTION BY USING SVM CLASSIFIER
numberTestImages = numel(imdsTest.Files)
for i = 1:numberTestImages
    imageTestUnresized = readimage(imdsTest,i);
    imageTest = imresize(imageTestUnresized,[128 64]);
    [featureVector,hogVisualization] = extractHOGFeatures(imageTest,'CellSize',cellSize);
    [prediction, scores] = predict(SVMModel,featureVector);
    figure(3);
    imshow(imageTest);
    title(strcat('Prediction:', string(prediction)))
end