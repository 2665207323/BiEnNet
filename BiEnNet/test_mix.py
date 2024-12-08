import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import dataset_test
import models.model
import pytorch_ssim


ssim_loss = pytorch_ssim.SSIM()
# 定义测试设置和路径
test_lowlight_images_path = "./dataset/Mix_data/"
model_checkpoint_path = "./weights/best_v1.pth"
output_dir = "./result/out_v1/mix"

# 检查文件夹是否存在，不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建测试数据集和数据加载器
test_dataset = dataset_test.test_dataset(
    low_img_dir=test_lowlight_images_path,
    high_img_dir=None,
    task="test_unpair",  # 对 成对测试图像 进行加载
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,  # 根据你的需求修改 batch_size
    shuffle=False,
    num_workers=0,  # 根据你的系统性能修改 num_workers
    pin_memory=True
)

# 创建模型实例并加载已训练的权重
model = models.model.BiEnNet().cuda()  # 根据你的模型配置修改参数
# 加载新的权重
model.load_state_dict(torch.load(model_checkpoint_path))
model.eval()

# 测试模型并保存结果
for iteration, (img_lowlight, high_exp_mean, img_name) in enumerate(test_loader):
    img_lowlight = img_lowlight.cuda()
    high_exp_mean = high_exp_mean.cuda()

    with torch.no_grad():
        start = time.time()
        enhanced_image, _ = model(img_lowlight, high_exp_mean)
        end_time = (time.time() - start)
        torch.cuda.empty_cache()  # 释放未使用的GPU内存, 防止内存累积，造成后面的大图片无法加载进去
        print("图像", img_name[0], "增强用时：", end_time)

    # 保存模型输出结果
    filename = os.path.basename(img_name[0])  # 对服务器上做适配，防止路径合并错误
    output_image_path = os.path.join(output_dir, filename)
    torchvision.utils.save_image(enhanced_image, output_image_path)

print("测试完成，结果已保存到:", output_dir)





