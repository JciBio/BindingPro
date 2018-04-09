digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
        'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
CountLabel = digitData.countEachLabel;
trainingNumFiles = 750;
rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(digitData, ...
				trainingNumFiles,'randomize');
% layers = [imageInputLayer([28 28 1])
%           convolution2dLayer(5,20)
%           reluLayer
%           maxPooling2dLayer(2,'Stride',2)
%           convolution2dLayer([4 3],12)
%           reluLayer
%           crossChannelNormalizationLayer(4)
%           maxPooling2dLayer(2,'Stride',2)
%           fullyConnectedLayer(256)
%           reluLayer
%           fullyConnectedLayer(10)
%           softmaxLayer
%           classificationLayer()];
% options = trainingOptions('sgdm','MaxEpochs',15, ...
% 	'InitialLearnRate',0.0001);
% convnet = trainNetwork(trainDigitData,layers,options);
% 
% layer='fc_2';
% trainVecs=activations(convnet,trainDigitData,layer);
% testVecs=activations(convnet,testDigitData,layer);
% trainLabels= trainDigitData.Labels;
% testLabels=testDigitData.Labels;
% 
% predictor=fitcecoc(trainVecs,trainLabels);
% YTest=predict(predictor,testVecs);
% accuracy=sum(YTest == testLabels)/numel(testLabels);
% disp(accuracy);
% plotconfusion(double(testLabels),double(YTest));
