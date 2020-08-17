%% EXTRACT HOG FEATURE VECTORS OF POSITIVE IMAGES
clc
array_img = zeros(length(imds.Files), 26244);
for i = 1:length(imds.Files)
    [featureVector,hogVisualization] = extractHOGFeatures(imread(imds.Files{i,1}));
    %[featureVector,hogVisualization] = extractHOGFeatures(img{i},'CellSize',[8 8]);
    size(featureVector)
    for j = 1:length(featureVector)
        array_img(i,j) = featureVector(1,j);
    end
end