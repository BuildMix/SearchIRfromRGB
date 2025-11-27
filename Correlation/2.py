import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# Part 1: 模型与预处理 (保持不变)
# ==========================================
class VGGBlock3Extractor(nn.Module):
    def __init__(self):
        super(VGGBlock3Extractor, self).__init__()
        full_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = full_vgg16.features[:17]
        self.inst_norm = nn.InstanceNorm2d(256, affine=False)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat = self.features(x)
        feat = self.inst_norm(feat)
        return feat

def apply_sobel_algorithm(gray_img):
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)
    return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"错误: 找不到文件 {image_path}")
        return None, None, None
    orig_img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    sobel_img = apply_sobel_algorithm(gray_img)
    sobel_rgb = cv2.merge([sobel_img, sobel_img, sobel_img])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(sobel_rgb).unsqueeze(0)
    return img_tensor, sobel_img, cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

# ==========================================
# Part 2: Step 3 相关度计算
# ==========================================
def correlation_layer(feat_rgb, feat_ir):
    B, C, H, W = feat_rgb.shape
    N = H * W
    feat_rgb_norm = F.normalize(feat_rgb, p=2, dim=1)
    feat_ir_norm = F.normalize(feat_ir, p=2, dim=1)
    feat_rgb_flat = feat_rgb_norm.view(B, C, N)
    feat_ir_flat = feat_ir_norm.view(B, C, N)
    correlation_matrix = torch.bmm(feat_rgb_flat.transpose(1, 2), feat_ir_flat)
    return correlation_matrix

# ==========================================
# Part 3: 可视化 (新增 N x N 矩阵展示)
# ==========================================
def visualize_query_and_matrix(corr_matrix, query_x, query_y, feat_size, orig_size, rgb_viz, ir_viz):
    """
    展示: 1. RGB查询点  2. IR热力图  3. N x N 全局矩阵
    """
    B, N, N_dim = corr_matrix.shape
    H_feat, W_feat = feat_size
    H_orig, W_orig = orig_size

    # --- 1. 准备局部热力图数据 ---
    fx = int(query_x * (W_feat / W_orig))
    fy = int(query_y * (H_feat / H_orig))
    fx = min(max(fx, 0), W_feat - 1)
    fy = min(max(fy, 0), H_feat - 1)
    query_idx = fy * W_feat + fx

    # 提取那一行 (Row)
    correlation_map_flat = corr_matrix[0, query_idx, :]
    correlation_map = correlation_map_flat.view(H_feat, W_feat).cpu().numpy()
    heatmap_resized = cv2.resize(correlation_map, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)
    
    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_resized)

    # --- 2. 准备 N x N 矩阵数据 ---
    # 直接获取完整的矩阵 (注意: 如果图片太大，这个矩阵显示起来可能会很慢)
    full_matrix = corr_matrix[0].cpu().numpy()

    # --- 3. 绘图 (三联) ---
    plt.figure(figsize=(18, 6))

    # [图1] RGB 查询点
    plt.subplot(1, 3, 1)
    plt.title(f"1. Query on Visible (x={query_x}, y={query_y})")
    plt.imshow(rgb_viz)
    plt.plot(query_x, query_y, 'r+', markersize=20, markeredgewidth=3)
    plt.axis('off')

    # [图2] IR 响应热力图
    plt.subplot(1, 3, 2)
    plt.title(f"2. Best Match on IR (Sim={max_val:.2f})")
    plt.imshow(ir_viz, cmap='gray')
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.6)
    plt.plot(max_loc[0], max_loc[1], 'go', markersize=15, markerfacecolor='none', markeredgewidth=3)
    plt.axis('off')

    # [图3] N x N 全局相关性矩阵
    plt.subplot(1, 3, 3)
    plt.title(f"3. Global Correlation Matrix ({N}x{N})")
    # aspect='auto' 确保矩阵填满正方形窗口，origin='upper' 确保坐标系正确
    im = plt.imshow(full_matrix, cmap='viridis', aspect='auto', origin='upper')
    plt.xlabel("Infrared Pixel Index (Flattened)")
    plt.ylabel("Visible Pixel Index (Flattened)")
    plt.colorbar(im, fraction=0.046, pad=0.04) # 给矩阵加个色条

    plt.tight_layout()
    plt.show()

# ==========================================
# 主流程
# ==========================================
def main():
    rgb_path = 'visible.png'
    ir_path = 'infrared.png'
    
    print("正在加载模型...")
    model = VGGBlock3Extractor()
    model.eval()

    print("Step 1: Sobel 预处理...")
    rgb_tensor, _, rgb_orig = process_image(rgb_path)
    ir_tensor, _, ir_orig = process_image(ir_path)
    
    if rgb_tensor is None or ir_tensor is None:
        return

    print("Step 2: VGG 特征提取...")
    with torch.no_grad():
        rgb_feat = model(rgb_tensor)
        ir_feat = model(ir_tensor)

    print("Step 3: 计算全图相关性矩阵...")
    corr_matrix = correlation_layer(rgb_feat, ir_feat)
    
    B, N, _ = corr_matrix.shape
    print(f"矩阵维度: {B} x {N} x {N} (共有 {N*N/1e6:.1f} 百万个匹配对)")

    # 交互：选择图像中心点
    H, W = rgb_orig.shape[:2]
    cx, cy = W // 2, H // 2

    print("正在生成三联可视化图...")
    visualize_query_and_matrix(
        corr_matrix, 
        cx, cy, 
        feat_size=(rgb_feat.shape[2], rgb_feat.shape[3]), 
        orig_size=(H, W), 
        rgb_viz=rgb_orig, 
        ir_viz=ir_orig
    )

if __name__ == "__main__":
    main()