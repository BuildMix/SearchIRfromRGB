import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 第一部分：核心模型定义 (保持不变)
# ==========================================
class VGGBlock3Extractor(nn.Module):
    def __init__(self):
        super(VGGBlock3Extractor, self).__init__()
        # 加载预训练权重
        full_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # 截取前3个Block (到 layer 16)
        self.features = full_vgg16.features[:17]
        # Instance Normalization (关键：去对比度)
        self.inst_norm = nn.InstanceNorm2d(256, affine=False)
        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat = self.features(x)
        feat = self.inst_norm(feat)
        return feat

# ==========================================
# 第二部分：融合预处理 (Sobel -> Tensor)
# ==========================================
def apply_sobel_algorithm(gray_img):
    """
    纯数学运算：输入灰度图数组，返回 Sobel 梯度图数组
    """
    # 1. 计算 Sobel (float64)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    # 2. 转回 uint8
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)
    # 3. 融合
    return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

def process_image_to_vgg_input(image_path):
    """
    读取原始图片 -> 转灰度 -> 算Sobel -> 转3通道 -> 转Tensor -> 归一化
    """
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 {image_path}")
        return None, None

    # 1. 读取原始图片
    # 无论原图是彩色的还是红外的，先读进来
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        return None, None

    # 2. 转灰度 (Sobel 需要灰度输入)
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    # 3. 执行 Sobel 算法 (在内存中进行，不保存图片)
    sobel_img = apply_sobel_algorithm(gray_img)

    # 4. 准备 VGG 输入
    # VGG 需要 3 通道 RGB，我们把单通道 Sobel 复制 3 份
    sobel_rgb = cv2.merge([sobel_img, sobel_img, sobel_img])

    # 5. Tensor 转换与归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 增加 Batch 维度: [3, H, W] -> [1, 3, H, W]
    img_tensor = transform(sobel_rgb).unsqueeze(0)

    # 返回 Tensor 用于计算，返回 sobel_img 用于最后画图展示
    return img_tensor, sobel_img

# ==========================================
# 第三部分：可视化辅助函数
# ==========================================
def compute_activation_maps(feat_tensor):
    """
    从 Feature Tensor 中计算 Mean 和 Max 激活图
    """
    feats = feat_tensor.squeeze(0).cpu().numpy()
    mean_act = np.mean(feats, axis=0)
    max_idx = np.argmax(np.mean(feats, axis=(1, 2)))
    max_act = feats[max_idx]
    return mean_act, max_act

# ==========================================
# 第四部分：主流程
# ==========================================
def main():
    # --- 配置 ---
    rgb_path = 'visible.png'     # 原始可见光图
    ir_path = 'infrared.png'     # 原始红外图
    output_filename = 'final_vgg_output.png'
    
    # 1. 初始化模型
    print("正在初始化 VGG 模型...")
    model = VGGBlock3Extractor()
    model.eval()
    
    # 2. 处理数据
    print("正在读取并预处理图片 (Sobel + Tensor转换)...")
    rgb_tensor, rgb_sobel_viz = process_image_to_vgg_input(rgb_path)
    ir_tensor, ir_sobel_viz = process_image_to_vgg_input(ir_path)
    
    if rgb_tensor is None or ir_tensor is None:
        return

    # 3. VGG 推理
    print("正在提取特征...")
    with torch.no_grad():
        rgb_feat = model(rgb_tensor)
        ir_feat = model(ir_tensor)

    # 4. 准备绘图数据
    print("正在生成最终结果...")
    rgb_mean, rgb_max = compute_activation_maps(rgb_feat)
    ir_mean, ir_max = compute_activation_maps(ir_feat)

    # 5. 绘制最终图表 (3行2列)
    plt.figure(figsize=(12, 12))

    # 第一行：Sobel 处理后的样子 (这是网络实际看到的"输入")
    plt.subplot(3, 2, 1)
    plt.title("Network Input (Visible Sobel)")
    plt.imshow(rgb_sobel_viz, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.title("Network Input (Infrared Sobel)")
    plt.imshow(ir_sobel_viz, cmap='gray')
    plt.axis('off')

    # 第二行：特征均值 (整体注意力)
    plt.subplot(3, 2, 3)
    plt.title("VGG Feature Mean (RGB)")
    plt.imshow(rgb_mean, cmap='inferno')
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.title("VGG Feature Mean (IR)")
    plt.imshow(ir_mean, cmap='inferno')
    plt.axis('off')

    # 第三行：最强特征通道 (显著纹理)
    plt.subplot(3, 2, 5)
    plt.title("Strongest Channel (RGB)")
    plt.imshow(rgb_max, cmap='viridis')
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.title("Strongest Channel (IR)")
    plt.imshow(ir_max, cmap='viridis')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"完成！最终结果已保存为: {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()