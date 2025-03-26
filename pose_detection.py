import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import json
import torch
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PoseDetector:
    def __init__(self, device: Optional[torch.device] = None):
        """初始化姿态检测器"""
        try:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device
                
            logger.info(f"使用设备: {self.device}")
            
            # 加载YOLOv8模型
            try:
                # 设置PyTorch的安全加载配置
                from ultralytics.nn.tasks import PoseModel
                torch.serialization.add_safe_globals([PoseModel])
                
                # 使用更稳定的配置初始化YOLO模型
                model_path = 'yolov8n-pose.pt'
                if not os.path.exists(model_path):
                    logger.info("下载YOLO模型...")
                    self.model = YOLO('yolov8n-pose.pt')
                else:
                    # 使用weights_only=False加载模型
                    self.model = YOLO(model_path, task='pose')
                
                # 设置模型配置
                self.model.conf = 0.25  # 置信度阈值
                self.model.iou = 0.45   # NMS IOU阈值
                self.model.agnostic_nms = True  # 类别无关NMS
                
                # 将模型移动到指定设备
                self.model.to(self.device)
                logger.info("YOLO模型加载成功")
                
            except Exception as e:
                logger.error(f"YOLO模型加载失败: {str(e)}")
                raise
            
            # COCO关键点定义
            self.coco_keypoints = {
                0: 'nose',
                1: 'left_eye',
                2: 'right_eye',
                3: 'left_ear',
                4: 'right_ear',
                5: 'left_shoulder',
                6: 'right_shoulder',
                7: 'left_elbow',
                8: 'right_elbow',
                9: 'left_wrist',
                10: 'right_wrist',
                11: 'left_hip',
                12: 'right_hip',
                13: 'left_knee',
                14: 'right_knee',
                15: 'left_ankle',
                16: 'right_ankle'
            }
            
            # 关键点映射（COCO到SMPL）
            self.keypoint_mapping = {
                5: 17,  # 左肩
                6: 18,  # 右肩
                7: 19,  # 左肘
                8: 20,  # 右肘
                9: 21,  # 左手腕
                10: 22, # 右手腕
                11: 23, # 左髋
                12: 24, # 右髋
                13: 25, # 左膝
                14: 26, # 右膝
                15: 27, # 左踝
                16: 28  # 右踝
            }
            
        except Exception as e:
            logger.error(f"初始化姿态检测器失败: {str(e)}")
            raise
    
    def normalize_keypoints(self, keypoints, image_shape):
        """将关键点坐标归一化到[-1, 1]范围"""
        h, w = image_shape[:2]
        normalized_keypoints = {}
        
        for idx, (x, y) in keypoints.items():
            # 将坐标原点移到图像中心
            x_centered = (x - w/2) / (w/2)
            y_centered = (y - h/2) / (h/2)
            normalized_keypoints[idx] = (x_centered, y_centered)
        
        return normalized_keypoints
    
    def process_video(self, video_path, output_dir):
        """处理视频文件，只进行姿态检测"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        # 准备存储数据
        pose_data = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 处理当前帧
                frame_data = self.process_frame(frame, frame_count, fps)
                if frame_data:
                    pose_data.append(frame_data)
                
                frame_count += 1
                
                # 检查是否按下'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # 保存姿态数据
        self.save_pose_data(pose_data, video_path, output_dir)
        
        return pose_data
    
    def process_frame(self, frame, frame_count, fps):
        """处理单帧图像，只进行姿态检测"""
        # 使用YOLOv8进行姿态检测
        results = self.model(frame, conf=0.25, verbose=False)
        
        if len(results) == 0:
            print(f"第{frame_count}帧未检测到人体姿态")
            return None
            
        # 获取所有检测到的人物的置信度和关键点
        detections = []
        for i, result in enumerate(results):
            # 检查是否有检测结果
            if len(result.boxes) == 0:
                continue
                
            confidence = result.boxes.conf[0].item()
            keypoints = result.keypoints.xy[0].cpu().numpy()
            detections.append({
                'confidence': confidence,
                'keypoints': keypoints
            })
        
        # 如果没有有效的检测结果
        if not detections:
            print(f"第{frame_count}帧没有有效的检测结果")
            return None
        
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 选择置信度最高的人物
        best_detection = detections[0]
        keypoints = best_detection['keypoints']
        
        # 提取所有COCO关键点
        coco_keypoints = {}
        for i in range(len(self.coco_keypoints)):
            if i < len(keypoints):
                x, y = keypoints[i][:2]
                if x > 0 and y > 0:
                    coco_keypoints[i] = (x, y)
        
        # 检查是否检测到足够的关键点
        if len(coco_keypoints) < 4:  # 至少需要4个关键点
            print(f"第{frame_count}帧检测到的关键点数量不足")
            return None
            
        # 归一化关键点坐标
        normalized_keypoints = self.normalize_keypoints(coco_keypoints, frame.shape)
        
        print(f"第{frame_count}帧检测到{len(coco_keypoints)}个关键点，置信度: {best_detection['confidence']:.2f}")
        
        # 可视化YOLO检测结果
        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv8 Pose Detection', annotated_frame)
        cv2.waitKey(1)  # 显示1毫秒
        
        return {
            'frame': frame_count,
            'time': frame_count / fps,
            'confidence': best_detection['confidence'],
            'keypoints': normalized_keypoints,
            'original_keypoints': coco_keypoints  # 保存原始坐标
        }
    
    def save_pose_data(self, pose_data, video_path, output_dir):
        """保存姿态数据到CSV文件"""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 准备CSV数据
        csv_data = []
        for data in pose_data:
            row = {
                'frame': data['frame'],
                'time': data['time'],
                'confidence': data['confidence']
            }
            for smpl_idx, (x, y) in data['keypoints'].items():
                row[f'kp_{smpl_idx}_x'] = x
                row[f'kp_{smpl_idx}_y'] = y
            csv_data.append(row)
        
        # 保存为CSV
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f"{base_name}_pose_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"姿态数据已保存到: {csv_path}")
        
        # 保存处理结果摘要
        summary = {
            'video_path': video_path,
            'total_frames': len(pose_data),
            'processed_frames': len([d for d in pose_data if d is not None])
        }
        
        with open(os.path.join(output_dir, f"{base_name}_detection_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

    def detect_video(self, video_path: str) -> Dict:
        """检测视频中的姿态"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"视频信息: {total_frames}帧, {fps}fps, {width}x{height}")
        
        # 初始化结果存储
        results = {
            'keypoints': [],
            'confidence': [],
            'metadata': {
                'total_frames': total_frames,
                'fps': fps,
                'width': width,
                'height': height
            }
        }
        
        # 处理每一帧
        with tqdm(total=total_frames, desc="检测姿态") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 检测姿态
                pose_results = self.model(frame, verbose=False)
                
                # 提取关键点和置信度
                if len(pose_results) > 0:
                    keypoints = pose_results[0].keypoints.data[0].cpu().numpy()
                    confidence = pose_results[0].keypoints.conf[0].cpu().numpy()
                    
                    results['keypoints'].append(keypoints)
                    results['confidence'].append(confidence)
                else:
                    # 如果没有检测到姿态，添加空数据
                    results['keypoints'].append(np.zeros((17, 3)))
                    results['confidence'].append(np.zeros(17))
                    
                pbar.update(1)
                
        cap.release()
        
        # 转换为numpy数组
        results['keypoints'] = np.array(results['keypoints'])
        results['confidence'] = np.array(results['confidence'])
        
        return results
        
    def save_results(self, results: Dict, output_file: str):
        """保存检测结果到HDF5文件"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # 保存关键点
            f.create_dataset('keypoints', data=results['keypoints'])
            
            # 保存置信度
            f.create_dataset('confidence', data=results['confidence'])
            
            # 保存元数据
            metadata = f.create_group('metadata')
            for key, value in results['metadata'].items():
                metadata.attrs[key] = value
                
        logger.info(f"结果已保存到: {output_file}")
        
    def load_results(self, input_file: str) -> Dict:
        """从HDF5文件加载检测结果"""
        results = {}
        
        with h5py.File(input_file, 'r') as f:
            # 加载关键点
            results['keypoints'] = f['keypoints'][:]
            
            # 加载置信度
            results['confidence'] = f['confidence'][:]
            
            # 加载元数据
            results['metadata'] = dict(f['metadata'].attrs)
            
        return results

def main():
    # 创建检测器实例
    detector = PoseDetector()
    
    # 处理视频
    video_root = "./videos"
    output_dir = "./output/detection"
    os.makedirs(output_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(video_root):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                print(f"\n处理视频: {video_path}")
                detector.process_video(video_path, output_dir)

if __name__ == "__main__":
    main() 