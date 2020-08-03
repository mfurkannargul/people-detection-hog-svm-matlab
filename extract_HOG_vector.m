%% EXTRACT HOG FEATURE VECTORS OF POSITIVE IMAGES
clc
array_img = zeros(length(img), 25000);
for i = 1:length(img)
    %[featureVector,hogVisualization] = extractHOGFeatures(img{i});
    [featureVector,hogVisualization] = extractHOGFeatures(img{i},'CellSize',[8 8]);
    featureVector;
    for j = 1:length(featureVector)
        array_img(i,j) = featureVector(1,j);
    end
end
array_img;
size(array_img)
array_img(1,:)

vector =  transpose(nonzeros((array_img(1,:))')); % can be used to extract the vector without zeros

