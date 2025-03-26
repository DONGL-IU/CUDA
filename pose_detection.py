import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
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
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_yolo_model()
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        
    def load_yolo_model(self) -> YOLO:
        """加载YOLO模型"""
        try:
            model_path = Path("models/yolo/yolov8n-pose.pt")
            if not model_path.exists():
                logger.info("下载YOLOv8姿态检测模型...")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model = YOLO("yolov8n-pose.pt")
                model.export(format="pt")
                logger.info("模型下载完成")
            else:
                model = YOLO(str(model_path))
            return model
        except Exception as e:
            logger.error(f"加载YOLO模型失败: {str(e)}")
            raise
    
    def detect_video(self, video_path: str, output_path: str) -> None:
        """处理视频文件并检测姿态"""
        try:
            logger.info(f"开始处理视频: {video_path}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            # 获取视频信息
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"视频信息: {frame_count}帧, {fps} FPS, {width}x{height}")
            
            # 初始化结果存储
            keypoints_data = []
            confidence_data = []
            
            # 创建进度条
            pbar = tqdm(total=frame_count, desc="处理帧")
            
            # 处理每一帧
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
                
                pbar.update(1)
            
            pbar.close()
            cap.release()
            
            # 转换为numpy数组
            keypoints_data = np.array(keypoints_data)
            confidence_data = np.array(confidence_data)
            
            # 保存结果
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('keypoints', data=keypoints_data)
                f.create_dataset('confidence', data=confidence_data)
                f.attrs['frame_count'] = frame_count
                f.attrs['fps'] = fps
                f.attrs['video_name'] = Path(video_path).stem
            
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
    # 创建姿态检测器实例
    detector = PoseDetector()
    
    # 处理视频目录
    video_dir = Path("./videos")
    output_dir = Path("./output/detection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理所有视频文件
    for video_path in video_dir.glob("*.mp4"):
        output_path = output_dir / f"{video_path.stem}_pose.h5"
        detector.detect_video(str(video_path), str(output_path))

if __name__ == "__main__":
    main() 