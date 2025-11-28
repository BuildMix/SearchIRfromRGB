import numpy as np
import math

def calculate_errors(orig_box_norm, trans_corners, img_w, img_h):
    """计算中心点误差和尺寸误差"""
    ncx, ncy, nw, nh = orig_box_norm
    gt_cx = ncx * img_w
    gt_cy = ncy * img_h
    gt_w = nw * img_w
    gt_h = nh * img_h
    
    pts = trans_corners.reshape(4, 2)
    pred_cx = np.mean(pts[:, 0])
    pred_cy = np.mean(pts[:, 1])
    
    top_w = np.linalg.norm(pts[1] - pts[0])
    bottom_w = np.linalg.norm(pts[2] - pts[3])
    pred_w = (top_w + bottom_w) / 2.0
    
    left_h = np.linalg.norm(pts[3] - pts[0])
    right_h = np.linalg.norm(pts[2] - pts[1])
    pred_h = (left_h + right_h) / 2.0
    
    center_error = math.sqrt((gt_cx - pred_cx)**2 + (gt_cy - pred_cy)**2)
    width_error = pred_w - gt_w
    height_error = pred_h - gt_h
    
    return {
        'gt_center': (gt_cx, gt_cy),
        'pred_center': (pred_cx, pred_cy),
        'gt_size': (gt_w, gt_h),
        'pred_size': (pred_w, pred_h),
        'center_error': center_error,
        'width_error': width_error,
        'height_error': height_error
    }