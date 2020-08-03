folder = 'C:\Users\furka\Desktop\Internship - Signalton Technology\MATLAB\HOG + SVM\positive';  
directory = dir(fullfile(folder,'*.jpg'));
for k = 1:numel(directory)
  filename = fullfile(folder,directory(k).name);
  img{k} = imread(filename);
end
length(img)
size(img)
