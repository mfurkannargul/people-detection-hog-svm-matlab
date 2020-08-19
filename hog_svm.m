%% LOAD TRAINING IMAGES
clc, clear
folder = 'C:\Users\furka\Documents\GitHub\people-detection-hog-svm-matlab\training_images';
imdsTrain = imageDatastore(folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
numberImages = numel(imdsTrain.Files);

%% RESIZE IMAGES & EXTRACT A SAMPLE HOG FEATURE VECTOR
k = 11160;
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
    image = imresize(img_unresized,[128 64]);
    size(image)
    [featureVector,hogVisualization] = extractHOGFeatures(image,'CellSize', cellSize);
    size(featureVector)
    for j = 1:length(hogFeatureSize)
        arrayTrainingFeatures(k,j) = featureVector(1,j);
    end
end
trainingLabels = imdsTrain.Labels;
size(trainingLabels)

%% TRAIN SVM CLASSIFIER
SVMModel = fitcecoc(arrayTrainingFeatures, trainingLabels);

%% TEST ON TRAINING IMAGES
for i = 10580:12170
    img1 = readimage(imdsTrain,i);
    hog1 = extractHOGFeatures(img,'CellSize',[2 2]);
    [pred, scores] = predict(SVMModel,hog1);
    figure(3);
    imshow(img1);
    title(strcat('Prediction:', string(pred)))
end
k = 12020;
img1 = readimage(imdsTrain,k);
hog1 = extractHOGFeatures(img,'CellSize',[2 2]);
[pred, scores] = predict(SVMModel,hog1);
figure(3);
imshow(img1);
title(strcat('Prediction:', string(pred)))