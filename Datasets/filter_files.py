import os
import argparse

def main(parent_folder, dry_run=True):
    # 定义三个子文件夹及其对应的文件扩展名
    subfolders = {
        "ir": [".png", ".jpg", ".jpeg", ".bmp"],  # 支持常见图片格式
        "vi": [".png", ".jpg", ".jpeg", ".bmp"],
        "labels": [".txt"]
    }
    
    # 遍历每个子文件夹
    for subfolder, extensions in subfolders.items():
        folder_path = os.path.join(parent_folder, subfolder)
        
        if not os.path.exists(folder_path):
            print(f"警告: 子文件夹不存在 - {folder_path}")
            continue
        
        print(f"\n处理文件夹: {folder_path}")
        kept_files = 0
        deleted_files = 0
        
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # 跳过目录
            if os.path.isdir(file_path):
                continue
            
            # 检查文件扩展名是否匹配
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                continue
            
            # 提取纯数字文件名（如"0000"）
            base_name = os.path.splitext(filename)[0]
            if not base_name.isdigit():
                print(f"  跳过非数字命名文件: {filename}")
                continue
            
            # 转换为整数序号
            try:
                file_num = int(base_name)
            except ValueError:
                print(f"  无效序号（跳过）: {filename}")
                continue
            
            # 保留能被60整除的文件
            if file_num % 60 == 0:
                kept_files += 1
            else:
                deleted_files += 1
                if dry_run:
                    print(f"  [模拟] 将删除: {filename} (序号={file_num})")
                else:
                    try:
                        os.remove(file_path)
                        print(f"  已删除: {filename} (序号={file_num})")
                    except Exception as e:
                        print(f"  删除失败 {filename}: {str(e)}")
        
        print(f"  保留文件数: {kept_files}")
        print(f"  删除文件数: {deleted_files} (模拟模式)" if dry_run else f"  删除文件数: {deleted_files}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='每60张图片/标签保留一张，删除其余文件')
    parser.add_argument('folder', type=str, help='包含ir/vi/labels的父文件夹路径')
    parser.add_argument('--execute', action='store_true', help='执行真实删除（默认仅模拟）')
    args = parser.parse_args()

    # 设置安全模式：默认不执行真实删除
    DRY_RUN = not args.execute
    
    if not os.path.exists(args.folder):
        print(f"错误: 指定路径不存在 - {args.folder}")
        exit(1)
    
    print(f"开始处理 (安全模式: {'开启' if DRY_RUN else '关闭'})")
    main(args.folder, dry_run=DRY_RUN)
    
    if DRY_RUN:
        print("\n提示: 以上为模拟操作。确认无误后添加 --execute 参数执行真实删除")