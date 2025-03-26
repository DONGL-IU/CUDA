import os
import torch
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union, TypeVar, Sequence
from tqdm import tqdm
import h5py
from ultralytics import YOLO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PoseDetector:
    def __init__(self, device: Optional[torch.device] = None):
        """初始化姿态检测器"""
        try:
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"使用设备: {self.device}")
            
            # 加载YOLO模型
            logger.info("加载YOLO模型...")
            self.model = YOLO('yolov8n-pose.pt')
            self.model.to(self.device)
            logger.info("YOLO模型加载成功")
            
        except Exception as e:
            logger.error(f"初始化姿态检测器失败: {str(e)}")
            raise
    
    def detect_video(self, video_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        """处理视频文件并检测姿态"""
        try:
            video_path = Path(video_path)
            output_path = Path(output_path)
            logger.info(f"开始处理视频: {video_path}")
            logger.info(f"输出路径: {output_path}")
            
            # 检查视频文件是否存在
            if not video_path.exists():
                logger.error(f"视频文件不存在: {video_path}")
                return
                
            # 打开视频文件
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            # 获取视频信息
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"视频信息: {frame_count}帧, {fps} FPS, {width}x{height}")
            
            # 初始化结果存储
            keypoints_data: List[np.ndarray] = []
            confidence_data: List[np.ndarray] = []
            
            # 创建进度条
            pbar = tqdm(total=frame_count, desc="处理帧")
            
            # 处理每一帧
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 检测姿态
                results = self.model(frame, verbose=False)
                
                # 提取关键点和置信度
                if len(results) > 0 and len(results[0].keypoints) > 0:
                    keypoints = results[0].keypoints[0].cpu().numpy()
                    confidence = results[0].keypoints.conf[0].cpu().numpy()
                    
                    keypoints_data.append(keypoints)
                    confidence_data.append(confidence)
                else:
                    # 如果没有检测到姿态，使用零填充
                    keypoints_data.append(np.zeros((17, 2)))
                    confidence_data.append(np.zeros(17))
                
                frame_idx += 1
                pbar.update(1)
                
                # 每100帧输出一次进度
                if frame_idx % 100 == 0:
                    logger.info(f"已处理 {frame_idx}/{frame_count} 帧")
            
            pbar.close()
            cap.release()
            
            # 转换为numpy数组
            keypoints_data = np.array(keypoints_data)
            confidence_data = np.array(confidence_data)
            
            # 保存结果
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('keypoints', data=keypoints_data)
                f.create_dataset('confidence', data=confidence_data)
                f.attrs['frame_count'] = frame_count
                f.attrs['fps'] = fps
                f.attrs['video_name'] = video_path.stem
                f.attrs['width'] = width
                f.attrs['height'] = height
            
            logger.info(f"姿态检测完成，结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"处理视频失败: {str(e)}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """处理单帧图像"""
        try:
            # 检测姿态
            results = self.model(frame, verbose=False)
            
            if len(results) > 0 and len(results[0].keypoints) > 0:
                keypoints = results[0].keypoints[0].cpu().numpy()
                confidence = results[0].keypoints.conf[0].cpu().numpy()
                return keypoints, confidence
            else:
                return np.zeros((17, 2)), np.zeros(17)
                
        except Exception as e:
            logger.error(f"处理帧失败: {str(e)}")
            return np.zeros((17, 2)), np.zeros(17)

def main():
    """主函数"""
    try:
        # 创建姿态检测器实例
        detector = PoseDetector()
        
        # 获取当前工作目录
        current_dir = Path.cwd()
        logger.info(f"当前工作目录: {current_dir}")
        
        # 处理视频目录
        video_dir = Path("/content/CUDA/videos")  # 使用正确的视频目录路径
        output_dir = current_dir / "output" / "detection"
        
        logger.info(f"视频目录: {video_dir}")
        logger.info(f"输出目录: {output_dir}")
        
        # 检查视频目录是否存在
        if not video_dir.exists():
            logger.error(f"视频目录不存在: {video_dir}")
            return
            
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有视频文件
        video_files = list(video_dir.glob("*.mp4"))
        logger.info(f"找到 {len(video_files)} 个视频文件")
        
        if not video_files:
            logger.error("没有找到视频文件")
            return
            
        # 处理所有视频文件
        for video_path in video_files:
            logger.info(f"开始处理视频: {video_path}")
            output_path = output_dir / f"{video_path.stem}_pose.h5"
            detector.detect_video(video_path, output_path)
            
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 