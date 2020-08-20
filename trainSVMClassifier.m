%% LOAD TRAINING IMAGES
clc, clear
folder = 'C:\Users\furka\Documents\GitHub\people-detection-hog-svm-matlab\training_images';
imdsTrain = imageDatastore(folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
numberImages = numel(imdsTrain.Files)

%% EXTRACT A SAMPLE HOG FEATURE VECTOR
k = 11161;
img_unresized = readimage(imdsTrain,k);
img = imresize(img_unresized,[128 64]);
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
    img = imresize(img_unresized,[128 64]);
    size(image)
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
for i = 10580:12170
    imgage_unresized = readimage(imdsTrain,i);
    image = imresize(imgage_unresized,[128 64]);
    [featureVector,hogVisualization] = extractHOGFeatures(image,'CellSize',cellSize);
    [prediction, scores] = predict(SVMModel,featureVector);
    figure(2);
    imshow(image);
    title(strcat('Prediction:', string(prediction)))
end

%% SAVE THE TRAINED CLASSIFIER FOR FURTHER USE
%save SVMModel
save('SVMModel', '-v7.3')