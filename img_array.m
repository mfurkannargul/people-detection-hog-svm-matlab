%% INSERT THE POSITIVE IMAGES TO AN ARRAY
clc
folder = 'C:\Users\furka\Documents\GitHub\people-detection-hog-svm-matlab\positive';  
trainingSet = dir(fullfile(folder,'*.jpg'));
for k = 1:numel(trainingSet)
  filename = fullfile(folder,trainingSet(k).name);
  img{k} = imread(filename);
end
length(img)
size(img)

