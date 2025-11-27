import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 定义孪生特征提取器 (Siamese Feature Extractor) ---
class VGGBlock3Extractor(nn.Module):
    def __init__(self):
        super(VGGBlock3Extractor, self).__init__()
        
        # 加载预训练的 VGG16
        # weights='DEFAULT' 会自动下载 ImageNet 的预训练权重
        full_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # --- 核心操作：网络切片 ---
        # VGG16 的 features 部分结构如下：
        # Block 1: layers 0-4   (64 channels)
        # Block 2: layers 5-9   (128 channels)
        # Block 3: layers 10-16 (256 channels) --> 我们截取到这里
        # layer 16 是 MaxPool2d
        self.features = full_vgg16.features[:17]
        
        # --- 核心操作：Instance Normalization ---
        # 这一步对于跨模态对齐至关重要，它抹平了 RGB 和 IR 的对比度差异
        # 256 是 Block 3 输出的通道数
        self.inst_norm = nn.InstanceNorm2d(256, affine=False)
        
        # 冻结参数 (我们现在只做推理/提取，不需要训练)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 1. 提取卷积特征
        feat = self.features(x)
        # 2. 应用实例归一化
        feat = self.inst_norm(feat)
        return feat

# --- 2. 图片预处理工具 ---
def preprocess_image(image_path):
    """
    读取 Sobel 图片并转换为 VGG 需要的 Tensor 格式
    """
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 {image_path}")
        return None

    # 读取刚才生成的 Sobel 结果图 (单通道灰度)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None

    # VGG 需要 3 通道输入 (RGB)。
    # 虽然我们的 Sobel 是灰度的，但我们将其重复 3 次以适应网络接口。
    img_rgb = cv2.merge([img_gray, img_gray, img_gray])

    # 定义转换：转 Tensor -> 标准化 (使用 ImageNet 的均值和方差)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 增加 Batch 维度: [3, H, W] -> [1, 3, H, W]
    img_tensor = transform(img_rgb).unsqueeze(0)
    return img_tensor, img_gray

# --- 3. 可视化工具 ---
def visualize_features(feat_tensor):
    """
    将 256 个通道的高维特征可视化。
    方法：计算所有通道的平均激活热力图，或者展示最强的通道。
    """
    # 去掉 Batch 维度: [1, 256, H, W] -> [256, H, W]
    feats = feat_tensor.squeeze(0).cpu().numpy()
    
    # 方法 A: 平均热力图 (看到整体关注点)
    mean_activation = np.mean(feats, axis=0)
    
    # 方法 B: 找出激活值最大的那个通道 (看到最强烈的特征)
    # 这通常代表网络找到的最显著的边缘或纹理
    max_idx = np.argmax(np.mean(feats, axis=(1, 2)))
    max_activation = feats[max_idx]
    
    return mean_activation, max_activation

# --- 4. 主函数 ---
def main():
    # --- 配置 ---
    # 这里的输入是你上一步 Sobel 代码的输出文件
    rgb_sobel_path = 'output_sobel_visible.png'
    ir_sobel_path = 'output_sobel_infrared.png'
    output_viz_filename = 'output_vgg_features.png'
    
    # 1. 初始化模型
    print("正在加载 VGG-16 模型 (首次运行会自动下载权重)...")
    model = VGGBlock3Extractor()
    model.eval() # 设置为评估模式
    
    # 2. 读取并预处理数据
    print("正在读取 Sobel 图像...")
    rgb_data = preprocess_image(rgb_sobel_path)
    ir_data = preprocess_image(ir_sobel_path)
    
    if rgb_data is None or ir_data is None:
        print("错误：无法读取输入图片，请先运行上一步的 Sobel 代码生成图片！")
        return

    rgb_tensor, rgb_orig = rgb_data
    ir_tensor, ir_orig = ir_data

    # 3. 特征提取 (Forward Pass)
    print("正在提取高维几何特征...")
    with torch.no_grad():
        rgb_feat = model(rgb_tensor)
        ir_feat = model(ir_tensor)

    print(f"特征提取完成！输出维度: {rgb_feat.shape}") 
    # 预期输出: [1, 256, H/8, W/8] (因为经过了3次 Pooling，尺寸缩小8倍)

    # 4. 可视化
    print("正在生成可视化对比图...")
    rgb_mean, rgb_max = visualize_features(rgb_feat)
    ir_mean, ir_max = visualize_features(ir_feat)

    # 绘制结果
    plt.figure(figsize=(14, 10))

    # 第一行：输入 (Sobel 图)
    plt.subplot(3, 2, 1)
    plt.title("Input A: Visible Sobel")
    plt.imshow(rgb_orig, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.title("Input B: Infrared Sobel")
    plt.imshow(ir_orig, cmap='gray')
    plt.axis('off')

    # 第二行：VGG 特征热力图 (平均激活)
    # 这代表网络认为"哪里有内容"
    plt.subplot(3, 2, 3)
    plt.title("VGG Feature (RGB, Mean Activation)")
    plt.imshow(rgb_mean, cmap='inferno') # 使用热力图色板
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.title("VGG Feature (IR, Mean Activation)")
    plt.imshow(ir_mean, cmap='inferno')
    plt.axis('off')

    # 第三行：VGG 最强特征通道
    # 这代表网络提取到的最显著的某种几何纹理
    plt.subplot(3, 2, 5)
    plt.title("VGG Strongest Channel (RGB)")
    plt.imshow(rgb_max, cmap='viridis')
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.title("VGG Strongest Channel (IR)")
    plt.imshow(ir_max, cmap='viridis')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_viz_filename, bbox_inches='tight', dpi=150)
    print(f"结果已保存至: {output_viz_filename}")
    plt.show()

if __name__ == "__main__":
    main()