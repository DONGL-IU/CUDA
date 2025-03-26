# SMPL 3D人体姿态重建项目

这个项目使用YOLOv8和SMPL模型来实现从视频中检测人体姿态并重建3D人体模型。

## 在Google Colab中运行

1. 首先，在Colab中克隆项目仓库：
```bash
!git clone https://github.com/your_username/your_repo.git
cd your_repo
```

2. 运行环境配置脚本：
```python
!python setup_colab.py
```

3. 激活Python 3.10虚拟环境：
```bash
source venv/bin/activate
```

4. 运行主程序：
```python
!python main.py --input "/content/videos" --output "/content/output_results" --no-drive
```

## 参数说明

- `--input`: 输入视频目录路径
- `--output`: 输出结果目录路径
- `--no-drive`: 不使用Google Drive（可选）
- `--batch-size`: 批处理大小（默认：1）
- `--device`: 运行设备（默认：自动选择）

## 输出目录结构

```
output/
├── detection/         # 2D姿态检测结果
├── reconstruction/    # 3D重建结果
├── visualization/     # 可视化结果
└── opensim/          # OpenSim模型文件
```

## 依赖包版本

- Python 3.10
- PyTorch 2.0.1
- NumPy 1.23.5
- OpenCV 4.7.0.72
- Pandas 1.5.3
- H5py 3.8.0
- Ultralytics 8.0.0
- SMPLX 0.1.28
- SciPy 1.10.1

## 注意事项

1. 确保在Google Colab中运行此项目
2. 使用Python 3.10以获得最佳兼容性
3. 如果需要使用GPU，请在Colab中选择GPU运行时
4. SMPL模型文件会在首次运行时自动下载

## 常见问题

Q: 遇到CUDA内存不足的问题怎么办？
A: 可以尝试减小batch_size或使用更小的输入视频分辨率。

Q: 如何在本地环境运行？
A: 建议使用conda创建Python 3.10环境，然后按照依赖包列表安装所需包。

## 许可证

本项目采用MIT许可证。详见LICENSE文件。
