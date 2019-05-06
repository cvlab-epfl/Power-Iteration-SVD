clear
clc
%% ResNet50 batch_size=32
% bn = [5.64, 5.94, 5.39];
% zca = [5.66, 5.63, 5.4, 5.38];
% pca = [5.77, 5.77, 5.87, 4.88];

% bn_sts = [mean(bn), std(bn)];
% 
% zca_sts = [mean(zca), std(zca)];
% 
% pca_sts = [mean(pca), std(pca)];


%% ResNet18 batch_size = 128
bn = [4.66, 5.05, 4.66, 4.85];
bn_sts = [mean(bn), std(bn)];

pca = [4.63, 4.75, 4.74, 4.78];
pca_sts = [mean(pca), std(pca)];

zca = [5.22, 5.04, 5.15, 4.96];
zca_sts = [mean(zca), std(zca)];
