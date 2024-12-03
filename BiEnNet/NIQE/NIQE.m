
% 指定测试文件夹的路径
testDir = fullfile('./test_data/');

% 获取测试文件夹中的所有.png图片文件
testFiles = dir(fullfile(testDir, '*.png'));

% 加载之前保存的模型
modelLoadPath = 'E:/G_disks/Matlab_workspace/NIQE/niqe_model.mat';
load(modelLoadPath, 'model');

% 初始化变量以累积NIQE分数和计数
totalNIQE = 0;
numImages = length(testFiles);

% 循环遍历每个测试图片并计算NIQE分数
for i = 1:length(testFiles)
    % 读取当前测试图片
    currentFile = fullfile(testDir, testFiles(i).name);
    I = imread(currentFile);
    
    % 计算NIQE分数并累积
    niqeI = niqe(I, model);
    totalNIQE = totalNIQE + niqeI;
    
    % 可选择将每个NIQE分数存储在一个数组中，以后进行分析或处理
    % niqeScores(i) = niqeI;
end

% 计算平均NIQE分数
averageNIQE = totalNIQE / numImages;
fprintf('Average NIQE score for %d images is %0.4f.\n', numImages, averageNIQE);