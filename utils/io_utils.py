# utils/io_utils.py
import os
from pathlib import Path

def load_yolo_labels(txt_path):
    """读取 YOLO 格式标签"""
    boxes = []
    if not os.path.exists(txt_path): return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # 忽略 class_id, 取 [cx, cy, w, h]
                box = [float(x) for x in parts[1:5]]
                boxes.append(box)
    return boxes

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """自动生成不重复的路径 (exp1, exp2...)"""
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                path = Path(p)
                break
    if mkdir: path.mkdir(parents=True, exist_ok=True)
    return str(path)

def append_results_to_summary(txt_path, img_id, results):
    """写入统计结果到 txt"""
    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write(f"\n[Image ID: {img_id}]\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Idx':<4}| {'GT Center':<14}| {'Pred Center':<14}| {'Diff':<7}| {'Size Diff'}\n")
        f.write("-" * 65 + "\n")
        
        total_err, count = 0.0, 0
        for i, item in enumerate(results):
            m = item.get('metrics')
            if m:
                diff = m['center_error']
                f.write(f"{i:<4}| ({m['gt_center'][0]:.0f},{m['gt_center'][1]:.0f})      | ({m['pred_center'][0]:.0f},{m['pred_center'][1]:.0f})      | {diff:.1f}   | ({m['width_error']:.1f}, {m['height_error']:.1f})\n")
                total_err += diff
                count += 1
        
        if count > 0: f.write(f"平均误差: {total_err/count:.2f} px\n")
        else: f.write("匹配失败\n")