import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math

def correlation_layer(feat_rgb, feat_ir):
    """计算特征图的相关性矩阵"""
    B, C, H, W = feat_rgb.shape
    N = H * W
    feat_rgb_norm = F.normalize(feat_rgb, p=2, dim=1)
    feat_ir_norm = F.normalize(feat_ir, p=2, dim=1)
    feat_rgb_flat = feat_rgb_norm.view(B, C, N)
    feat_ir_flat = feat_ir_norm.view(B, C, N)
    correlation_matrix = torch.bmm(feat_rgb_flat.transpose(1, 2), feat_ir_flat)
    return correlation_matrix

def ransac_smart(corr_matrix, norm_box, sobel_map, feat_size, orig_size):
    """
    核心匹配逻辑：基于相关性和RANSAC计算变换矩阵
    """
    B, N, N_dim = corr_matrix.shape
    H_feat, W_feat = feat_size
    H_orig, W_orig = orig_size
    
    # 解析归一化框
    ncx, ncy, nw, nh = norm_box
    pcx, pcy = int(ncx * W_orig), int(ncy * H_orig)
    pw, ph = int(nw * W_orig), int(nh * H_orig)
    x_min, y_min = pcx - pw//2, pcy - ph//2
    x_max, y_max = pcx + pw//2, pcy + ph//2

    # 生成网格点
    grid_size = 6
    xs = np.linspace(x_min + pw*0.1, x_max - pw*0.1, grid_size)
    ys = np.linspace(y_min + ph*0.1, y_max - ph*0.1, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points_rgb = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    valid_matches_src = []
    valid_matches_dst = []
    
    sobel_threshold = 15 
    search_radius = W_orig // 3 #设定搜索半径为原始图像宽度的 1/3
    feat_search_radius = search_radius * (W_feat / W_orig)

    # 匹配点筛选
    for pt in points_rgb:
        px, py = int(pt[0]), int(pt[1])
        if px < 0 or px >= W_orig or py < 0 or py >= H_orig: continue
        if sobel_map[py, px] < sobel_threshold: continue
            
        fx = int(px * (W_feat / W_orig))
        fy = int(py * (H_feat / H_orig))
        fx, fy = min(max(fx, 0), W_feat - 1), min(max(fy, 0), H_feat - 1)
        query_idx = fy * W_feat + fx
        
        corr_row = corr_matrix[0, query_idx, :]
        
        # 局部掩码搜索最大值
        mask = torch.ones_like(corr_row) * float('-inf')
        win_x_min = max(0, int(fx - feat_search_radius))
        win_x_max = min(W_feat, int(fx + feat_search_radius))
        win_y_min = max(0, int(fy - feat_search_radius))
        win_y_max = min(H_feat, int(fy + feat_search_radius))
        
        mask_2d = mask.view(H_feat, W_feat)
        mask_2d[win_y_min:win_y_max, win_x_min:win_x_max] = 0
        masked_corr = corr_row + mask_2d.view(-1)
        
        max_idx = torch.argmax(masked_corr).item()
        match_fy = max_idx // W_feat
        match_fx = max_idx % W_feat
        match_px = int(match_fx * (W_orig / W_feat))
        match_py = int(match_fy * (H_orig / H_feat))
        
        valid_matches_src.append([px, py])
        valid_matches_dst.append([match_px, match_py])

    valid_matches_src = np.array(valid_matches_src, dtype=np.float32)
    valid_matches_dst = np.array(valid_matches_dst, dtype=np.float32)
    
    if len(valid_matches_src) < 3: return None, None

    # 计算仿射变换
    M_affine, inliers = cv2.estimateAffinePartial2D(
        valid_matches_src, valid_matches_dst, 
        method=cv2.RANSAC, ransacReprojThreshold=10.0
    )
    
    # 结果校验与兜底策略
    use_fallback = True
    final_M = None

    if M_affine is not None:
        a = M_affine[0, 0]
        b = M_affine[1, 0]
        scale = math.sqrt(a**2 + b**2)
        if 0.8 <= scale <= 1.2:
            final_M = M_affine
            use_fallback = False
    
    if use_fallback:
        deltas = valid_matches_dst - valid_matches_src
        dx = np.median(deltas[:, 0])
        dy = np.median(deltas[:, 1])
        final_M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    # 变换原始框坐标
    box_corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(box_corners, final_M)
    
    return transformed_corners, inliers