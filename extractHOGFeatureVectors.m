%% RESIZE IMAGES AND TAKE HOG FEATURE VECTORS
clc, clear
folder = 'C:\Users\furka\Documents\GitHub\people-detection-hog-svm-matlab\training_images';
imds = imageDatastore(folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
inputSize = [128 128];
% imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
array_img = zeros(length(imds.Files), 8100);
% trainingSet = dir(fullfile(imds,'*.jpg'));
for k = 1:numel(imds.Files)
    img{k} = imread(imds.Files{k,1});
    img{k} = imresize(img{k},inputSize);
    imshow(img{k})
    [featureVector,hogVisualization] = extractHOGFeatures(img{k});
    size(featureVector)
    for j = 1:length(featureVector)
        array_img(k,j) = featureVector(1,j);
    end
end