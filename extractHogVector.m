%% EXTRACT HOG FEATURE VECTORS OF POSITIVE IMAGES
clc
array_img = zeros(length(img), 8100);
for i = 1:length(img)
    [featureVector,hogVisualization] = extractHOGFeatures(img{i});
    %[featureVector,hogVisualization] = extractHOGFeatures(img{i},'CellSize',[8 8]);
    size(featureVector)
    for j = 1:length(featureVector)
        array_img(i,j) = featureVector(1,j);
    end
end