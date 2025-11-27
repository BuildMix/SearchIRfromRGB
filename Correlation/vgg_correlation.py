import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# Part 1: 模型与预处理 (沿用之前确定的方案)
# ==========================================
class VGGBlock3Extractor(nn.Module):
    def __init__(self):
        super(VGGBlock3Extractor, self).__init__()
        # 加载 VGG16 权重
        full_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # 截取前 17 层 (前3个Block)
        self.features = full_vgg16.features[:17]
        # Instance Normalization (去对比度)
        self.inst_norm = nn.InstanceNorm2d(256, affine=False)
        # 冻结参数
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat = self.features(x)
        feat = self.inst_norm(feat)
        return feat

def apply_sobel_algorithm(gray_img):
    """ 计算 Sobel 梯度图 """
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)
    return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

def process_image(image_path):
    """ 读取 -> 灰度 -> Sobel -> Tensor """
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 {image_path}")
        return None, None, None

    # 读取原始图片
    orig_img = cv2.imread(image_path)
    # 转灰度用于 Sobel
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    # 计算 Sobel
    sobel_img = apply_sobel_algorithm(gray_img)
    # 转回 3 通道伪彩色 (适应 VGG 输入)
    sobel_rgb = cv2.merge([sobel_img, sobel_img, sobel_img])

    # 归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(sobel_rgb).unsqueeze(0)
    
    # 返回: Tensor(计算用), Sobel图(展示用), 原图(展示用)
    return img_tensor, sobel_img, cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

# ==========================================
# Part 2: Step 3 相关度计算 (新增核心部分)
# ==========================================
def correlation_layer(feat_rgb, feat_ir):
    """
    输入: [B, C, H, W]
    输出: [B, H*W, H*W] 相关性矩阵
    """
    B, C, H, W = feat_rgb.shape
    N = H * W
    
    # 1. L2 归一化 (在通道维度 C 上)
    # 这一步把特征向量变成单位向量，只保留"方向"(几何结构)信息
    feat_rgb_norm = F.normalize(feat_rgb, p=2, dim=1)
    feat_ir_norm = F.normalize(feat_ir, p=2, dim=1)
    
    # 2. 空间展平 (Flatten)
    # [B, C, H, W] -> [B, C, N]
    feat_rgb_flat = feat_rgb_norm.view(B, C, N)
    feat_ir_flat = feat_ir_norm.view(B, C, N)
    
    # 3. 矩阵乘法 (Matrix Multiplication)
    # 计算所有像素对的点积相似度
    # [B, N, C] * [B, C, N] -> [B, N, N]
    correlation_matrix = torch.bmm(feat_rgb_flat.transpose(1, 2), feat_ir_flat)
    
    return correlation_matrix

# ==========================================
# Part 3: 可视化查询 (新增核心部分)
# ==========================================
def visualize_query(corr_matrix, query_x, query_y, feat_size, orig_size, rgb_viz, ir_viz):
    """
    在 RGB 图上选一个点，画出它在 IR 图上的热力图
    """
    B, N, N = corr_matrix.shape
    H_feat, W_feat = feat_size
    H_orig, W_orig = orig_size

    # 1. 将原图坐标映射到特征图坐标 (缩小8倍)
    # query_x, query_y 是原图上的坐标
    fx = int(query_x * (W_feat / W_orig))
    fy = int(query_y * (H_feat / H_orig))
    
    # 边界保护
    fx = min(max(fx, 0), W_feat - 1)
    fy = min(max(fy, 0), H_feat - 1)

    # 2. 计算特征图上的扁平化索引 (Index)
    query_idx = fy * W_feat + fx

    # 3. 从相关性矩阵中提取这一行
    # 这一行包含了该点与 IR 图上所有点的相似度
    # shape: [1, N] -> [N]
    correlation_map_flat = corr_matrix[0, query_idx, :]

    # 4. 变回二维图片形状
    # shape: [H_feat, W_feat]
    correlation_map = correlation_map_flat.view(H_feat, W_feat).cpu().numpy()

    # 5. 上采样回原图大小 (为了好看)
    # 使用双线性插值平滑放大
    heatmap_resized = cv2.resize(correlation_map, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)

    # --- 绘图 ---
    plt.figure(figsize=(12, 6))

    # 左图：RGB 查询图
    plt.subplot(1, 2, 1)
    plt.title(f"Query Point on Visible (x={query_x}, y={query_y})")
    plt.imshow(rgb_viz)
    # 画一个红色的十字标记查询点
    plt.plot(query_x, query_y, 'r+', markersize=20, markeredgewidth=3)
    plt.axis('off')

    # 右图：IR 响应热力图
    plt.subplot(1, 2, 2)
    plt.title("Correlation Heatmap on Infrared")
    # 先画底图
    plt.imshow(ir_viz, cmap='gray')
    # 再叠加半透明热力图 (alpha=0.6)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.6)
    plt.axis('off')

    # 自动寻找最匹配的点画个圈 (Argmax)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_resized)
    plt.plot(max_loc[0], max_loc[1], 'go', markersize=15, markerfacecolor='none', markeredgewidth=3)
    print(f"查询点: ({query_x}, {query_y}) -> 最佳匹配点: {max_loc}, 相似度: {max_val:.3f}")

    plt.tight_layout()
    plt.show()

# ==========================================
# 主流程
# ==========================================
def main():
    rgb_path = 'visible.png'
    ir_path = 'infrared.png'
    
    # 1. 初始化
    print("正在加载模型...")
    model = VGGBlock3Extractor()
    model.eval()

    # 2. 预处理 (Step 1)
    print("正在处理 Step 1 (Sobel)...")
    rgb_tensor, rgb_sobel, rgb_orig = process_image(rgb_path)
    ir_tensor, ir_sobel, ir_orig = process_image(ir_path)
    
    if rgb_tensor is None or ir_tensor is None:
        return

    # 3. 特征提取 (Step 2)
    print("正在处理 Step 2 (VGG Feature)...")
    with torch.no_grad():
        rgb_feat = model(rgb_tensor) # [1, 256, H', W']
        ir_feat = model(ir_tensor)

    # 4. 相关度计算 (Step 3)
    print("正在处理 Step 3 (Correlation Matrix)...")
    corr_matrix = correlation_layer(rgb_feat, ir_feat)
    print(f"相关性矩阵维度: {corr_matrix.shape}")
    # 预期: [1, N, N], 例如 [1, 12288, 12288]

    # 5. 可视化交互
    # 我们选择图片的中心点作为查询点来演示
    H_orig, W_orig = rgb_orig.shape[:2]
    center_x, center_y = W_orig // 2, H_orig // 2
    
    print(f"正在生成热力图 (查询点: 图像中心 {center_x}, {center_y})...")
    visualize_query(
        corr_matrix, 
        query_x=center_x, query_y=center_y, 
        feat_size=(rgb_feat.shape[2], rgb_feat.shape[3]), 
        orig_size=(H_orig, W_orig), 
        rgb_viz=rgb_orig, # 这里展示原图，方便看物体
        ir_viz=ir_orig
    )

if __name__ == "__main__":
    main()