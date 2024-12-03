% 读取训练文件
setDir = fullfile('E:/G_disks/Matlab_workspace/NIQE/train_data/');
imds = imageDatastore(setDir,'FileExtensions',{'.png'});

% 根据自己模型的训练参考图像 来 训练评估标准
model = fitniqe(imds);

% 保存训练好的模型到指定文件夹
modelSavePath = 'E:/G_disks/Matlab_workspace/niqe_model.mat';
save(modelSavePath, 'model');