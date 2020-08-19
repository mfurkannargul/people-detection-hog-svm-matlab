folder = 'C:\Users\furka\Documents\GitHub\people-detection-hog-svm-matlab\test_images';
inputSize = [128 128];

testSet = imageDatastore(folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
array_img1 = zeros(length(testSet.Files), 8100);
for k = 1:numel(testSet.Files)
    img1{k} = imread(testSet.Files{k,1});
    img1{k} = imresize(img1{k},inputSize);
    imshow(img1{k})
    [featureVector,hogVisualization] = extractHOGFeatures(img1{k});
    size(featureVector)
    for j = 1:length(featureVector)
        array_img1(k,j) = featureVector(1,j);
    end
end
CompactSVMModel = compact(Mdl)
whos('SVMModel','CompactSVMModel')

CompactSVMModel = fitPosterior(CompactSVMModel,...
    X,Y)

Z = array_img1;
size(array_img1)

[labels,PostProbs] = predict(CompactSVMModel,Z)
