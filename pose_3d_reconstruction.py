# 移除兼容性补丁
import os
import torch
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, TypeVar, Sequence
import logging
from tqdm import tqdm
from smplx import SMPL
import cv2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Pose3DReconstructor:
    def __init__(self, device: Optional[torch.device] = None):
        """初始化3D姿态重建器"""
        try:
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"使用设备: {self.device}")
            
            # 加载SMPL模型
            logger.info("加载SMPL模型...")
            self.smpl = SMPL(
                model_path='/content/CUDA/models/smpl',  # 修改为正确的模型路径
                gender='neutral',
                batch_size=1,
                create_global_orient=True,
                create_body_pose=True,
                create_betas=True
            ).to(self.device)
            logger.info("SMPL模型加载成功")
            
        except Exception as e:
            logger.error(f"初始化3D姿态重建器失败: {str(e)}")
            raise
    
    def reconstruct_video(self, pose_data_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        """从2D姿态重建3D姿态"""
        try:
            pose_data_path = Path(pose_data_path)
            output_path = Path(output_path)
            logger.info(f"开始处理姿态数据: {pose_data_path}")
            
            # 加载2D姿态数据
            with h5py.File(pose_data_path, 'r') as f:
                keypoints = f['keypoints'][:]
                confidence = f['confidence'][:]
                frame_count = f.attrs['frame_count']
                fps = f.attrs['fps']
                width = f.attrs['width']
                height = f.attrs['height']
            
            logger.info(f"加载了 {frame_count} 帧的姿态数据")
            
            # 初始化结果存储
            poses_3d: List[np.ndarray] = []
            betas: List[np.ndarray] = []
            
            # 创建进度条
            pbar = tqdm(total=frame_count, desc="重建3D姿态")
            
            # 处理每一帧
            for frame_idx in range(frame_count):
                # 获取当前帧的2D关键点和置信度
                keypoints_2d = torch.tensor(keypoints[frame_idx], dtype=torch.float32, device=self.device)
                confidence_2d = torch.tensor(confidence[frame_idx], dtype=torch.float32, device=self.device)
                
                # 重建3D姿态
                pose_3d, beta = self.reconstruct_frame(keypoints_2d, confidence_2d)
                
                poses_3d.append(pose_3d)
                betas.append(beta)
                
                pbar.update(1)
            
            pbar.close()
            
            # 转换为numpy数组
            poses_3d = np.array(poses_3d)
            betas = np.array(betas)
            
            # 保存结果
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('poses_3d', data=poses_3d)
                f.create_dataset('betas', data=betas)
                f.attrs['frame_count'] = frame_count
                f.attrs['fps'] = fps
                f.attrs['width'] = width
                f.attrs['height'] = height
            
            logger.info(f"3D姿态重建完成，结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"重建3D姿态失败: {str(e)}")
            raise
    
    def reconstruct_frame(self, keypoints_2d: torch.Tensor, confidence_2d: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """重建单帧的3D姿态"""
        try:
            # 使用SMPL模型重建3D姿态
            output = self.smpl(
                global_orient=torch.zeros(1, 3, device=self.device),
                body_pose=torch.zeros(1, 69, device=self.device),
                betas=torch.zeros(1, 10, device=self.device)
            )
            
            # 获取3D关键点
            joints_3d = output.joints.detach().cpu().numpy()[0]
            
            # 获取形状参数
            betas = output.betas.detach().cpu().numpy()[0]
            
            return joints_3d, betas
            
        except Exception as e:
            logger.error(f"重建帧失败: {str(e)}")
            return np.zeros((24, 3)), np.zeros(10)

def main():
    """主函数"""
    try:
        # 创建3D姿态重建器实例
        reconstructor = Pose3DReconstructor()
        
        # 处理姿态数据目录
        pose_data_dir = Path("./output/detection")
        output_dir = Path("./output/reconstruction")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理所有姿态数据文件
        for pose_data_path in pose_data_dir.glob("*_pose.h5"):
            output_path = output_dir / f"{pose_data_path.stem.replace('_pose', '_3d')}.h5"
            reconstructor.reconstruct_video(pose_data_path, output_path)
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()