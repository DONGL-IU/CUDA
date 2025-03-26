# Google Colab运行指南

## 1. 环境准备

### 1.1 创建新的Colab笔记本
1. 打开[Google Colab](https://colab.research.google.com/)
2. 创建新的笔记本
3. 确保使用GPU运行时：
   - 点击"运行时" -> "更改运行时类型"
   - 选择"GPU"作为硬件加速器

### 1.2 安装依赖
```python
# 克隆项目仓库
!git clone https://github.com/your-username/CUDA_SMPL.git
%cd CUDA_SMPL

# 安装依赖包
!pip install -r requirements.txt

# 安装额外需要的包
!pip install google-colab
```

### 1.3 下载SMPL模型
```python
# 创建模型目录
!mkdir -p models/smpl

# 从Google Drive下载SMPL模型（需要先上传到您的Google Drive）
from google.colab import drive
drive.mount('/content/drive')

# 复制模型文件
!cp /content/drive/MyDrive/smpl_models/* models/smpl/
```

## 2. 运行示例

### 2.1 上传视频文件
```python
# 上传视频文件
from google.colab import files
uploaded = files.upload()
video_path = list(uploaded.keys())[0]  # 获取上传的文件名
```

### 2.2 运行处理流程
```python
# 导入必要的模块
import os
from main import SMPLPipeline
import torch

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建输出目录
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 初始化处理流水线
pipeline = SMPLPipeline(device=device)

# 处理视频
pipeline.process_video(video_path, output_dir)
```

### 2.3 可视化结果
```python
# 显示处理后的3D模型
from IPython.display import Image
Image(filename='output/visualization/smpl_visualization.png')

# 显示关节角度
Image(filename='output/visualization/joint_angles.png')
```

## 3. 完整示例代码

```python
# 设置环境
!git clone https://github.com/your-username/CUDA_SMPL.git
%cd CUDA_SMPL
!pip install -r requirements.txt
!pip install google-colab

# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 创建必要的目录
!mkdir -p models/smpl
!mkdir -p output

# 复制SMPL模型文件
!cp /content/drive/MyDrive/smpl_models/* models/smpl/

# 上传视频文件
from google.colab import files
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# 导入必要的模块
import os
from main import SMPLPipeline
import torch

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 初始化并运行流水线
pipeline = SMPLPipeline(device=device)
pipeline.process_video(video_path, "output")

# 显示结果
from IPython.display import Image
Image(filename='output/visualization/smpl_visualization.png')
Image(filename='output/visualization/joint_angles.png')
```

## 4. 注意事项

1. GPU使用：
   - Colab的GPU是免费的，但使用时间有限
   - 建议在处理大型视频时保存中间结果
   - 可以使用Google Drive存储结果

2. 内存管理：
   - Colab的GPU内存是有限的
   - 处理大型视频时注意监控内存使用
   - 必要时可以分块处理视频

3. 文件存储：
   - 使用Google Drive存储大型文件
   - 定期清理临时文件
   - 保存重要的中间结果

4. 运行时间：
   - 免费版Colab有运行时间限制
   - 建议使用Pro版本处理大型项目
   - 可以设置自动保存检查点

## 5. 常见问题解决

1. GPU内存不足：
```python
# 清理GPU内存
import torch
torch.cuda.empty_cache()
```

2. 文件上传失败：
```python
# 使用Google Drive上传大文件
from google.colab import drive
drive.mount('/content/drive')
```

3. 模型加载失败：
```python
# 检查模型文件路径
import os
print(os.listdir('models/smpl'))
```

4. 运行中断：
```python
# 保存检查点
import pickle
with open('checkpoint.pkl', 'wb') as f:
    pickle.dump(pipeline.state_dict(), f)
``` 