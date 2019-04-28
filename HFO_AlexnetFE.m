% FEATURE EXTRATION USING ALEXNET
% Load images from dataset
%{
unzip('Dataset.zip');
imdsTrain = imageDatastore('Dataset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
unzip('testdata.zip');
imdsValidation = imageDatastore('testdata', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%}
unzip('Dataset_2.zip');
imds = imageDatastore('Dataset_2', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%{
imdsValidation.Labels = categorical({'Ripple','Not HFO','Not HFO','Not HFO','Ripple','Ripple','Ripple','Ripple','Ripple','Not HFO','Not HFO','Not HFO',...
'Not HFO','Ripple','Ripple','Not HFO','Not HFO','Not HFO','Ripple','Not HFO','Not HFO','Ripple','Ripple','Ripple',...
'Ripple','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Fast Ripple','Fast Ripple',...
'Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO',...
'Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Ripple','Ripple','Ripple','Not HFO','Not HFO',...
'Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Not HFO','Ripple','Ripple and Fast Ripple',...
'Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple','Ripple','Ripple and Fast Ripple','Ripple','Ripple','Ripple and Fast Ripple','Ripple','Not HFO','Not HFO',...
'Not HFO','Not HFO','Not HFO','Ripple and Fast Ripple','Not HFO','Ripple','Ripple','Ripple','Not HFO','Not HFO','Ripple','Ripple','Ripple',...
'Ripple','Ripple','Ripple','Not HFO','Not HFO','Not HFO','Ripple','Ripple','Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple',...
'Ripple','Not HFO','Ripple','Ripple','Ripple','Ripple','Ripple','Ripple','Not HFO','Not HFO','Ripple','Ripple',...
'Ripple','Ripple and Fast Ripple','Not HFO','Ripple','Ripple','Ripple','Ripple','Ripple','Ripple','Ripple','Ripple',...
'Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple','Ripple and Fast Ripple',...
'Not HFO','Not HFO','Not HFO','Ripple','Ripple','Ripple','Not HFO','Ripple and Fast Ripple','Ripple and Fast Ripple','Not HFO','Not HFO',...
'Ripple','Not HFO'}');
    %}
% Separate dataset into training and validation sets
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

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
analyzeNetwork(net);
% Display network architecture
net.Layers

% Get input image size information
inputSize = net.Layers(1).InputSize;

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsValidation);

layer = 'fc6';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsValidation.Labels;

classifier = fitcecoc(featuresTrain,YTrain);

YPred = predict(classifier,featuresTest);

idx = [1 10 25 100];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end






accuracy = mean(YPred == YTest)