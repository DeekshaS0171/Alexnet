% TRANSFER LEARNING USING ALEXNET
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

% Display an interactive visualization of the network architecture
% and detailed information about the network layers.
analyzeNetwork(net);

% Get input image size information
inputSize = net.Layers(1).InputSize;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODIFY PRETRAINED NN
%                             NOTE!!
% At this point I already modified the pretrained NN to have it do 4
% classifications for the purposes of classifying: HFO, Ripple, Fast
% Ripple, and Not HFO using the 'deepNetworkDesigner'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%  Extract all layers, except the last three, from the pretrained network
layersTransfer = net.Layers(1:end-3);

% Transfer the layers to the new classification task by replacing the last
% three layers with a fully connected layer, a softmax layer, and a classification
% output layer. Specify the options of the new fully connected layer according to
% the new data. Set the fully connected layer to have the same size as the number
% of classes in the new data. To learn faster in the new layers than in the
% transferred layers, increase the WeightLearnRateFactor and BiasLearnRateFactor
% values of the fully connected layer.

% This line should output # of classes
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODIFY SIZE OF INPUT IMAGES/DATA AUGMENTATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Translate pics up to 30 pixels horizontally/vertically
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

% Augment training images with imageAugmenter parameter
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

% Augment validation images with imageAugmenter parameter
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRAIN THE NETWORK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train the network that consists of the transferred and new layers.
 netTransfer = trainNetwork(augimdsTrain,layers,options);


%                               ^
%                               |
%                               |
%                               |

% AT THIS POINT THE NETWORK HAS BEEN TRAINED. YOU CAN COMMENT OUT
% 'netTransfer' SO YOU DON'T RETRAIN THE NETWORK WHEN DOING THE VALIDATION
% PORTION



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLASSIFY VALIDATION IMAGES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Classify the validation images using the fine-tuned network.
[YPred,scores] = classify(netTransfer,augimdsValidation);

% Display a number of sample validation images with their predicted labels.
idx = randperm(numel(imdsValidation.Files),16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end