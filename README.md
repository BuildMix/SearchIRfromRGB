# High-Resolution Sparse SCN Registration

这是一个基于 PyTorch 实现的高分辨率稀疏特征配准系统，专门用于\*\*可见光（Visible）**与**红外（Infrared）\*\*跨模态图像的目标框配准。

本项目通过提取浅层高分辨率特征（VGG Block1），结合 Sobel 梯度筛选与改进的 Smart RANSAC 算法，实现了在纹理差异巨大的跨模态图像对中进行鲁棒的目标位置校正。

## ✨ 主要特性

  * **高分辨率特征提取**：利用 VGG16 前 5 层提取特征，保留物体精细结构，并通过 `InstanceNorm` 消除光照/对比度差异。
  * **稀疏梯度筛选**：引入 Sobel 算子预处理，自动过滤平滑/无纹理区域，仅在强梯度区域进行特征采样，大幅提升匹配效率与精度。
  * **Smart RANSAC**：
      * 优先尝试计算仿射变换（Affine Partial 2D）。
      * 内置“熔断机制”：检测缩放系数是否在合理范围（0.8 - 1.2）。
      * 自动降级策略：若仿射变换不稳定，自动回退到中值流（Median Flow）纯平移模式。
  * **模块化设计**：代码结构清晰，解耦了模型、算法与业务逻辑，易于二次开发。

## 📂 项目结构

```text
├── core/                   # 核心算法模块
│   ├── feature_extractor.py # VGG 特征提取网络定义
│   ├── image_process.py     # 图像读取、Sobel 预处理、Tensor 转换
│   └── matcher.py           # 稀疏匹配逻辑与 Smart RANSAC 实现
│
├── utils/                  # 工具模块
│   ├── io_utils.py          # 文件 IO、标签读取、日志记录
│   └── visualizer.py        # 结果可视化绘制、误差计算
│
├── Datasets/               # 数据集目录 (需自行创建)
│   ├── vi/                  # 可见光图像
│   ├── ir/                  # 红外图像
│   └── labels/              # YOLO 格式标签文件
│
├── main.py                 # 程序入口
├── requirements.txt        # 依赖库列表
└── README.md               # 项目说明文档
```

## 🛠️ 环境依赖

请确保安装以下依赖库：

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
```

**建议版本：**

  * Python \>= 3.8
  * PyTorch \>= 1.10 (支持 CUDA 加速推荐)

## 🚀 快速开始

### 1\. 数据准备

请按照以下结构组织你的数据。图像文件名需一一对应（扩展名支持 `.jpg`/`.png`）。

  * **`Datasets/vi/`**: 存放可见光原图。
  * **`Datasets/ir/`**: 存放红外原图（与可见光图配对）。
  * **`Datasets/labels/`**: 存放 YOLO 格式的标签文件（对应可见光图的标注）。

> **YOLO 标签格式说明**:
> `<class_id> <x_center> <y_center> <width> <height>` (全部为 0-1 归一化数值)

### 2\. 运行程序

在项目根目录下运行：

```bash
python main.py
```

### 3\. 查看结果

程序运行结束后，结果将保存在 `runs/exp{N}` 目录下：

  * **`results.txt`**: 包含每个图像、每个目标框的中心点误差（像素）及尺寸误差统计。
  * **`*_result.png`**: 可视化对比图（左侧为可见光带原框，右侧为红外图带预测框）。

## 🧠 算法原理

### 1\. 预处理 (Gradient Filter)

为了防止在红外图像的平坦区域（如天空、路面）产生误匹配，系统首先计算图像的 Sobel 梯度图。在后续的特征采样阶段，只有梯度值大于阈值（默认为 15）的点才会被作为特征点。

### 2\. 特征匹配 (Sparse Matching)

  * **提取器**: VGG16 (Layers 0-4) -\> InstanceNorm。
  * **匹配策略**:
      * 在 RGB 图像的目标框内生成稀疏网格点。
      * 对每个网格点，提取 `1x1x64` 的特征向量。
      * 在 IR 特征图的对应局部邻域内，使用矩阵乘法（Dot Product）计算热力图。
      * 取热力图最大响应点作为匹配点。

### 3\. 几何校正 (Geometric Verification)

使用 `cv2.estimateAffinePartial2D` 计算变换矩阵。系统会检查计算出的缩放因子 `scale`：

  * 如果 `0.8 <= scale <= 1.2`，接受该仿射变换。
  * 否则，认为特征匹配噪声过大，强制使用 `Median Flow` 方法仅计算平移量（dx, dy），锁定缩放为 1.0。

## 📊 结果示例

运行日志 `results.txt` 示例：

```text
[Image ID: 000123]
-----------------------------------------------------------------
Idx | GT Center     | Pred Center   | Diff   | Size Diff
-----------------------------------------------------------------
0   | (320, 240)    | (325, 242)    | 5.4    | (1.2, -0.5)
1   | (100, 150)    | (102, 148)    | 2.8    | (0.1, 0.2)
平均误差: 4.10 px
```

## 📝 待办事项 / 改进方向

  - [ ] 继续提升性能。
  - [ ] 添加 ResNet 或 LoFTR 等更先进的特征提取器选项。
  - [ ] 尝试优化处理流程，减少限制。

-----

**License**
MIT © 2024