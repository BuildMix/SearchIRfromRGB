# core/matcher.py
import torch
import numpy as np
import cv2
import math

def sparse_ransac_smart(feat_rgb, feat_ir_flat, norm_box, sobel_map, feat_size, orig_size):
    """
    执行稀疏特征匹配与 RANSAC 几何校正
    """
    B, C, H_feat, W_feat = feat_rgb.shape
    H_orig, W_orig = orig_size
    
    # --- A. 解析 YOLO 框 ---
    ncx, ncy, nw, nh = norm_box
    pcx, pcy = int(ncx * W_orig), int(ncy * H_orig)
    pw, ph = int(nw * W_orig), int(nh * H_orig)
    x_min, y_min = pcx - pw//2, pcy - ph//2
    x_max, y_max = pcx + pw//2, pcy + ph//2

    # --- B. 网格采样 (6x6) ---
    grid_size = 6
    xs = np.linspace(x_min + pw*0.1, x_max - pw*0.1, grid_size)
    ys = np.linspace(y_min + ph*0.1, y_max - ph*0.1, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points_rgb = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    valid_matches_src = []
    valid_matches_dst = []
    
    # 参数设置
    sobel_threshold = 15          
    search_radius = W_orig // 3   
    feat_search_radius = search_radius * (W_feat / W_orig)

    # --- C. 稀疏循环匹配 ---
    for pt in points_rgb:
        px, py = int(pt[0]), int(pt[1])
        
        if px < 0 or px >= W_orig or py < 0 or py >= H_orig: continue
        if sobel_map[py, px] < sobel_threshold: continue # 梯度筛选
            
        # 映射坐标到特征层
        fx = int(px * (W_feat / W_orig))
        fy = int(py * (H_feat / H_orig))
        fx, fy = min(max(fx, 0), W_feat - 1), min(max(fy, 0), H_feat - 1)
        
        query_vec = feat_rgb[:, :, fy, fx].view(1, C, 1)
        
        # 矩阵乘法计算相似度
        heatmap_flat = torch.bmm(feat_ir_flat.transpose(1, 2), query_vec).flatten()
        
        # 空间掩码
        mask = torch.ones_like(heatmap_flat) * float('-inf')
        win_x_min = max(0, int(fx - feat_search_radius))
        win_x_max = min(W_feat, int(fx + feat_search_radius))
        win_y_min = max(0, int(fy - feat_search_radius))
        win_y_max = min(H_feat, int(fy + feat_search_radius))
        
        mask_2d = mask.view(H_feat, W_feat)
        mask_2d[win_y_min:win_y_max, win_x_min:win_x_max] = 0
        
        masked_heatmap = heatmap_flat + mask_2d.view(-1)
        max_idx = torch.argmax(masked_heatmap).item()
        
        match_fy = max_idx // W_feat
        match_fx = max_idx % W_feat
        match_px = int(match_fx * (W_orig / W_feat))
        match_py = int(match_fy * (H_orig / H_feat))
        
        valid_matches_src.append([px, py])
        valid_matches_dst.append([match_px, match_py])

    valid_matches_src = np.array(valid_matches_src, dtype=np.float32)
    valid_matches_dst = np.array(valid_matches_dst, dtype=np.float32)
    
    if len(valid_matches_src) < 3: return None, None

    # --- D. 鲁棒几何解算 (Smart RANSAC) ---
    M_affine, inliers = cv2.estimateAffinePartial2D(
        valid_matches_src, valid_matches_dst, 
        method=cv2.RANSAC, ransacReprojThreshold=10.0
    )
    
    use_fallback = True
    final_M = None

    if M_affine is not None:
        a, b = M_affine[0, 0], M_affine[1, 0]
        scale = math.sqrt(a**2 + b**2)
        if 0.8 <= scale <= 1.2:
            final_M = M_affine
            use_fallback = False
    
    if use_fallback:
        deltas = valid_matches_dst - valid_matches_src
        dx = np.median(deltas[:, 0])
        dy = np.median(deltas[:, 1])
        final_M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    # --- E. 变换原始框 ---
    box_corners = np.array([
        [x_min, y_min], [x_max, y_min], 
        [x_max, y_max], [x_min, y_max]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    transformed_corners = cv2.transform(box_corners, final_M)
    
    return transformed_corners, inliers