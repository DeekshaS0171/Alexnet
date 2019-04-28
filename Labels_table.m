%auimds = augmentedImageDatastore(outputSize,tbl) 
% FEATURE EXTRATION USING ALEXNET
% Load images from dataset
unzip('Dataset.zip');
imds = imageDatastore('Dataset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Separate dataset into training and validation sets
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

% Display some sample images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,25);
figure
for i = 1:25
    subplot(5,5,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

% Load Pretrained Network (Alexnet)
net = alexnet;

% Display network architecture
net.Layers

% Get input image size information
%inputSize = net.Layers(1).InputSize;
inputSize = [227,227,3]
augimdsTrain = augmentedImageDatastore(inputSize,'D:\MATLAB_D\HFO\Alextnet HFO\Raw photos\Labels.xlsx',imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsValidation);

layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsValidation.Labels;

classifier = fitcecoc(featuresTrain,YTrain);

YPred = predict(classifier,featuresTest);

idx = [1 5 10 15 20 25 30 35 3 6 100];
figure
for i = 1:numel(idx)
    subplot(2,5,i)
    I = readimage(imdsValidation,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

accuracy = mean(YPred == YTest)