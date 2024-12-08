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
test_lowlight_images_path = "./dataset/LOL_v1/eval15/low/"
test_highlight_images_path = "./dataset/LOL_v1/eval15/high/"
# test_lowlight_images_path = "./dataset/LOL_v2/eval15/low/"
# test_highlight_images_path = "./dataset/LOL_v2/eval15/high/"
model_checkpoint_path = "./weights/best_v1.pth"
output_dir = "./result/out_v1/lolv1/"

# 检查文件夹是否存在，不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建测试数据集和数据加载器
test_dataset = dataset_test.test_dataset(
    low_img_dir=test_lowlight_images_path,
    high_img_dir=test_highlight_images_path,
    task="test_pair",  # 对 成对测试图像 进行加载
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
model.load_state_dict(torch.load(model_checkpoint_path))  # 加载的权重
model.eval()


PSNR = 0
SSIM = 0
sum_time = 0
# 测试模型并保存结果
for iteration, (img_lowlight, img_highlight, high_exp_mean, img_name) in enumerate(test_loader):
    img_lowlight = img_lowlight.cuda()
    img_highlight = img_highlight.cuda()
    high_exp_mean = high_exp_mean.cuda()

    # 在模型中传递数据
    with torch.no_grad():
        start = time.time()
        enhanced_image, low_csn = model(img_lowlight, high_exp_mean)
        end_time = (time.time() - start)
        sum_time = sum_time + end_time
        print("图像增强用时：", end_time)

    imdff = enhanced_image - img_highlight
    rmse = torch.mean(imdff ** 2)
    Loss_psnr = 10 * torch.log10(1 / rmse)
    Loss_ssim = ssim_loss(enhanced_image, img_highlight)

    print("PSNR - ", Loss_psnr, "SSIM - ", Loss_ssim)
    PSNR += Loss_psnr
    SSIM += Loss_ssim

    # 保存模型输出结果
    filename = os.path.basename(img_name[0])  # 对服务器上做适配，防止路径合并错误
    output_image_path = os.path.join(output_dir, filename)
    torchvision.utils.save_image(low_csn, output_image_path)

print("PSNR_MEAN: ", PSNR/dataset_test.test_dataset.__len__(test_dataset), "SSIM_MEAN: ", SSIM/dataset_test.test_dataset.__len__(test_dataset))
print(dataset_test.test_dataset.__len__(test_dataset), "张图像的增强总用时为：", sum_time)
print("测试完成，结果已保存到:", output_dir)




