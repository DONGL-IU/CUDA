# SMPL处理流水线

这是一个基于CUDA加速的SMPL（Skinned Multi-Person Linear Model）处理流水线，用于从2D视频生成3D人体模型并进行物理模拟。

## 功能特点

- 基于YOLOv8的17个骨骼点目标检测
- 基于目标检测结果的SMPL模型转换
- 基于SMPL的骨骼点建模和物理模拟
- 支持CUDA加速
- 使用HDF5格式统一存储数据
- 提供可视化功能

## 项目结构

```
.
├── main.py                 # 主程序入口
├── pose_detection.py       # 姿态检测模块（YOLOv8）
├── pose_3d_reconstruction.py  # 3D重建模块（SMPL）
├── parallel_merge_results.py  # 数据合并模块
├── smpl_physics_model.py   # 物理模型模块（PyBullet）
├── requirements.txt        # 项目依赖
└── README.md              # 项目说明文档
```

## 数据存储架构

所有数据使用HDF5格式存储，具体结构如下：

### 1. 姿态检测结果
```h5
/keypoints: shape=(frames, 17, 2)  # 2D关键点坐标
/confidence: shape=(frames, 17)     # 关键点置信度
/attrs
    frame_count: int
    fps: float
    video_name: str
```

### 2. 3D重建结果
```h5
/vertices: shape=(frames, 6890, 3)  # SMPL顶点
/joints: shape=(frames, 24, 3)      # 关节点
/global_orient: shape=(frames, 3)    # 全局方向
/body_pose: shape=(frames, 69)       # 身体姿态
/betas: shape=(frames, 10)           # 形状参数
/transl: shape=(frames, 3)           # 平移
/attrs
    frame_count: int
    fps: float
    video_name: str
```

### 3. 物理模拟结果
```h5
/vertices: shape=(frames, 6890, 3)   # 更新后的顶点
/joints: shape=(frames, 24, 3)       # 更新后的关节点
/global_orient: shape=(frames, 3)     # 更新后的全局方向
/body_pose: shape=(frames, 69)        # 更新后的身体姿态
/betas: shape=(frames, 10)            # 形状参数
/transl: shape=(frames, 3)            # 更新后的平移
/physics_params
    /frame_0
        /position: shape=(3,)
        /velocity: shape=(3,)
        /angular_velocity: shape=(3,)
    /frame_1
        ...
```

## 安装说明

1. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载SMPL模型：
- 从[SMPL官网](https://smpl.is.tue.mpg.de/)下载SMPL模型文件
- 将模型文件放在`models/smpl/`目录下

## 使用方法

### 处理单个视频
```bash
python main.py --input your_video.mp4 --output output_directory --visualize
```

### 处理整个目录
```bash
python main.py --input input_directory --output output_directory --visualize
```

### 参数说明
- `--input`: 输入视频文件或目录
- `--output`: 输出目录
- `--visualize`: 启用可视化（可选）
- `--device`: 指定设备（cuda/cpu，默认自动选择）

## 性能优化

1. GPU内存优化：
- 使用批处理减少内存占用
- 及时释放不需要的张量
- 使用`torch.cuda.empty_cache()`清理GPU缓存

2. 存储优化：
- 使用HDF5格式压缩存储
- 分块处理大型数据集
- 使用内存映射处理大文件

3. 计算优化：
- 使用CUDA加速计算
- 多线程并行处理
- 使用缓存减少重复计算

## 注意事项

1. GPU要求：
- 建议使用NVIDIA GPU，显存至少8GB
- 支持CUDA 11.0或更高版本

2. 存储要求：
- 确保有足够的磁盘空间存储中间结果
- 建议使用SSD以提高读写速度

3. 内存要求：
- 建议系统内存至少16GB
- 处理大型视频时可能需要更多内存

## 错误处理

1. 常见错误：
- CUDA内存不足：减小批处理大小
- 文件读写错误：检查文件权限和磁盘空间
- 模型加载失败：检查模型文件路径

2. 调试方法：
- 查看日志文件了解详细错误信息
- 使用`--visualize`参数检查中间结果
- 检查数据格式是否符合要求

## 许可证

MIT License
