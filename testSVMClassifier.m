clc
%% LOAD SVM CLASSIFIER TRAINED
load SVMModel

%% LOAD TEST IMAGES
folderTest = 'C:\Users\furka\Documents\GitHub\people-detection-hog-svm-matlab\testImages';
imdsTest = imageDatastore(folderTest, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% DISPLAY A SAMPLE TEST IMAGE
t = 1;
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
    [prediction, scores] = predict(SVMModel,featureVector)
    figure(4);
    imshow(imageTest);
    title(strcat('Prediction:', string(prediction)))
end

%% INITIALIZE VARIABLES FOR SLIDING DETECTION WINDOW
stepSize = 5;
width = 100;
height = 40;
[y x] = size(imageTest)
location = zeros(y,x);
y = floor((y-height)/stepSize)+1;
x = floor((x-width)/stepSize)+1;

%% IMPLEMENT DETECTION WINDOW
y1 = 1;
x1 = 1;
for i = 1:y
    for j = 1:x
        image = imageTest(x1:x1+39,y1:y1+99);
        image = imresize(image,[128 64]);
        [featureVector,hogVisualization] = extractHOGFeatures(image,'CellSize',[2 2]);
        [prediction,scores] = predict(SVMModel,featureVector);
        if (string(prediction) == 'positive')
            pause(1);
            location(x1,y1) = -scores(2);
        end
        y1 = y1 + stepSize
    end
    y1 = 1;
    x1 = x1 + stepSize
end

%% DISPLAY DETECTION IN BOUNDING BOX
[y1 x1] = find(position > 0.75);
figure(6);
imshow(testImage)
for i = 1:length(y1)
    pause(1);
    imrect(gca, [x1(i),y1(i),100,40])
end
