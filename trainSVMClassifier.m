%% LOAD TRAINING IMAGES
clc, clear
folder = 'trainImages';
imdsTrain = imageDatastore(folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
numberImages = numel(imdsTrain.Files)

%% EXTRACT A SAMPLE HOG FEATURE VECTOR
k = 200;
img_unresized = readimage(imdsTrain,k);
img_resized = imresize(img_unresized,[128 64]);
img = rgb2gray(img_resized);
%img = edge(img_gray);
%img = img_unresized;
imshow(img);
[featureVector,hogVisualization] = extractHOGFeatures(img,'CellSize',[2 2]);
figure(1);
imshow(img);
hold on;
plot(hogVisualization)
title('hogVisualization')

%% DETERMINE CELL SIZE AND FEATURE SIZE
cellSize = [2 2]
hogFeatureSize = length(featureVector)

%% EXTRACT HOG FEATURE VECTORS OF TRAIN SET
arrayTrainingFeatures = zeros(numberImages,hogFeatureSize,'single');
for k = 1:numberImages
    img_unresized = readimage(imdsTrain,k);
    img_resized = imresize(img_unresized,[128 64]);
    img = rgb2gray(img_resized);
    %img = edge(img_gray);
    %img = img_unresized;
    size(img)
    [featureVector,hogVisualization] = extractHOGFeatures(img,'CellSize', cellSize);
    size(featureVector)
    for j = 1:length(hogFeatureSize)
        arrayTrainingFeatures(k,j) = featureVector(1,j);
    end
end
trainingLabels = imdsTrain.Labels;
size(trainingLabels)

%% TRAIN SVM CLASSIFIER
SVMModel = fitcsvm(arrayTrainingFeatures, trainingLabels)

%% TEST ON TRAINING IMAGES
start = 1;
correctPrediction = 0;
for i = start:numberImages
    image_unresized = readimage(imdsTrain,i);
    image_resized = imresize(image_unresized,[128 64]);
    image = rgb2gray(image_resized);
    %image = edge(img_gray);
    %image = imgage_unresized;
    [featureVector,hogVisualization] = extractHOGFeatures(image,'CellSize',cellSize);
    [prediction, scores] = predict(SVMModel,featureVector)
    figure(2);
    imshow(image_resized);
    title(strcat('Prediction:', string(prediction), '     Label:', string(trainingLabels(i))))
    if (string(prediction) == string(trainingLabels(i)))
        correctPrediction = correctPrediction + 1;
    end
    accuracy = correctPrediction / (i - start + 1) * 100  
end

%% PREDICTION ACCURACY
finalAccuracy = correctPrediction / (numberImages - start + 1) * 100

%% SAVE THE TRAINED CLASSIFIER FOR FURTHER USE
%save SVMModel
save('SVMModel', '-v7.3') % 134 pos, 158 neg, 65%