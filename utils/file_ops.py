import os
from pathlib import Path

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """自增路径（exp1, exp2...）"""
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                path = Path(p)
                break
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)

def load_yolo_labels(txt_path):
    """加载YOLO格式标签"""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO格式: class x y w h
                box = [float(x) for x in parts[1:5]]
                boxes.append(box)
    return boxes

def append_results_to_summary(txt_path, img_id, results):
    """将结果写入汇总txt"""
    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write(f"\n[Image ID: {img_id}]\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Idx':<5}| {'GT Center':<16}| {'Pred Center':<16}| {'Diff(px)':<9}| {'Size Diff(W,H)'}\n")
        f.write("-" * 60 + "\n")
        
        total_error = 0.0
        valid_count = 0
        
        for i, item in enumerate(results):
            metrics = item.get('metrics')
            if metrics is None: continue
            
            gt_cx, gt_cy = metrics['gt_center']
            pred_cx, pred_cy = metrics['pred_center']
            diff = metrics['center_error']
            w_diff = metrics['width_error']
            h_diff = metrics['height_error']
            
            str_gt = f"({int(gt_cx)}, {int(gt_cy)})"
            str_pred = f"({int(pred_cx)}, {int(pred_cy)})"
            str_diff = f"{diff:.2f}"
            str_size = f"({w_diff:.1f}, {h_diff:.1f})"
            
            f.write(f"{i:<5}| {str_gt:<16}| {str_pred:<16}| {str_diff:<9}| {str_size}\n")
            
            total_error += diff
            valid_count += 1
            
        f.write("=" * 60 + "\n")
        if valid_count > 0:
            avg_error = total_error / valid_count
            f.write(f"平均中心点误差: {avg_error:.2f} 像素\n")
        else:
            f.write("本图无有效匹配结果。\n")
        f.write("\n")