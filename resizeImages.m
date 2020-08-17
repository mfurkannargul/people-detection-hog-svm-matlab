%% INSERT THE POSITIVE IMAGES TO AN ARRAY AND RESIZE THEM
clc, clear
%folder1 = 'C:\Users\furka\Documents\GitHub\people-detection-hog-svm-matlab';

% images = imageDatastore(folder,'IncludeSubfolders');
% imds = imageDatastore(folder, ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');
% inputSize = [128 128];
% imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
% folder = 'C:\Users\furka\Desktop\pos';
% imds = imageDatastore(folder, ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');
folder = 'C:\Users\furka\Documents\GitHub\people-detection-hog-svm-matlab';
imds = imageDatastore(folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

inputSize = [128 128];
% imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);

% trainingSet = dir(fullfile(imds,'*.jpg'));
for k = 1:numel(imds.Files)
%     filename = fullfile(folder,trainingSet(k).name);
    img{k} = imread(imds.Files{k,1});
    img{k} = imresize(img{k},inputSize);
    imshow(img{k});
end