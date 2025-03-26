# 兼容性补丁（必须在所有其他导入之前）
import sys
import inspect

# 解决Python 3.11+中inspect.getargspec移除的问题
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# 解决torch._six兼容性问题
if sys.version_info >= (3, 11):
    import torch
    if hasattr(torch, '_six'):
        torch._six.PY3 = True
        torch._six.PY37 = False

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from smplx import SMPL
import torch.nn.functional as F
import json
from scipy.spatial.transform import Rotation
import threading
from queue import Queue
import concurrent.futures
from tqdm import tqdm
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import smplx

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_function_args(func):
    """获取函数参数的兼容性包装器"""
    try:
        sig = inspect.signature(func)
        return {
            'args': list(sig.parameters.keys()),
            'defaults': tuple(
                p.default for p in sig.parameters.values()
                if p.default is not p.empty
            )
        }
    except Exception as e:
        logger.warning(f"无法获取函数参数信息: {str(e)}")
        return {'args': [], 'defaults': ()}

class Pose3DReconstructor:
    def __init__(self, device: Optional[torch.device] = None):
        """初始化3D重建器"""
        try:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device
                
            logger.info(f"使用设备: {self.device}")
            
            # 加载SMPL模型
            self.load_smpl_model()
            
            # 创建输出目录
            os.makedirs("output/visualization", exist_ok=True)
            os.makedirs("output/opensim", exist_ok=True)
            
            # 初始化坐标缓存
            self.coordinate_cache = {}
            self.coordinate_threshold = 0.01  # 坐标差异阈值
            
            # 初始化线程锁
            self.cache_lock = threading.Lock()
            self.file_lock = threading.Lock()
            
            # 设置线程数
            self.num_threads = min(8, os.cpu_count() or 1)  # 最多使用8个线程
            
        except Exception as e:
            logger.error(f"初始化3D重建器失败: {str(e)}")
            raise
    
    def load_smpl_model(self):
        """加载SMPL模型"""
        try:
            model_path = 'models/smpl/SMPL_NEUTRAL.pkl'
            if not os.path.exists(model_path):
                logger.error(f"SMPL模型文件不存在: {model_path}")
                raise FileNotFoundError(f"SMPL模型文件不存在: {model_path}")
            
            # 检查Python版本兼容性
            if sys.version_info >= (3, 11):
                logger.info("检测到Python 3.11+，使用兼容性配置加载SMPL模型")
                # 使用更稳定的配置初始化SMPL模型
                self.model = smplx.create(
                    model_path=model_path,
                    model_type='smpl',
                    gender='neutral',
                    use_pca=False,
                    batch_size=1,
                    create_global_orient=True,
                    create_body_pose=True,
                    create_betas=True,
                    create_transl=True
                ).to(self.device)
            else:
                # 标准配置加载
                self.model = smplx.create(
                    model_path=model_path,
                    model_type='smpl',
                    gender='neutral',
                    use_pca=False,
                    batch_size=1
                ).to(self.device)
            
            logger.info("SMPL模型加载成功")
            
        except Exception as e:
            logger.error(f"SMPL模型加载失败: {str(e)}")
            if 'getargspec' in str(e):
                logger.error("检测到Python 3.11+兼容性问题，尝试应用补丁...")
                # 应用兼容性补丁
                if not hasattr(inspect, 'getargspec'):
                    inspect.getargspec = inspect.getfullargspec
                # 重新尝试加载
                self.model = smplx.create(
                    model_path=model_path,
                    model_type='smpl',
                    gender='neutral',
                    use_pca=False,
                    batch_size=1
                ).to(self.device)
                logger.info("通过兼容性补丁成功加载SMPL模型")
            else:
                raise
    
    def are_coordinates_similar(self, coords1, coords2):
        """检查两组坐标是否相似"""
        if len(coords1) != len(coords2):
            return False
        for k in coords1:
            if k not in coords2:
                return False
            x1, y1 = coords1[k]
            x2, y2 = coords2[k]
            if abs(x1 - x2) > self.coordinate_threshold or abs(y1 - y2) > self.coordinate_threshold:
                return False
        return True
    
    def get_cached_result(self, keypoints):
        """从缓存中获取结果"""
        keypoints_str = str(sorted(keypoints.items()))
        if keypoints_str in self.coordinate_cache:
            return self.coordinate_cache[keypoints_str]
        return None
    
    def cache_result(self, keypoints, result):
        """缓存结果"""
        keypoints_str = str(sorted(keypoints.items()))
        self.coordinate_cache[keypoints_str] = result
    
    def process_chunk(self, chunk_data, video_path, output_dir, chunk_id):
        """处理数据块"""
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
        
        # 设置视频帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_data[0]['frame'])
        
        # 创建块进度条
        chunk_pbar = tqdm(chunk_data, 
                         desc=f"处理块 {chunk_id + 1}", 
                         leave=False,
                         position=chunk_id + 1)
        
        # 处理块中的每一帧
        for data in chunk_pbar:
            frame_count = data['frame']
            ret, frame = cap.read()
            if not ret:
                break
            
            # 更新进度条描述
            chunk_pbar.set_description(f"处理块 {chunk_id + 1} - 帧 {frame_count}")
            
            # 检查关键点数据
            keypoints = {}
            for i in range(17):
                x_key = f'kp_{i}_x'
                y_key = f'kp_{i}_y'
                if x_key in data and y_key in data:
                    x, y = data[x_key], data[y_key]
                    if not np.isnan(x) and not np.isnan(y):
                        keypoints[i] = (float(x), float(y))
            
            # 如果关键点数量不足，跳过该帧
            if len(keypoints) < 4:
                chunk_pbar.set_postfix({"状态": "跳过 - 关键点不足"})
                continue
            
            # 检查缓存
            with self.cache_lock:
                cached_result = self.get_cached_result(keypoints)
            
            if cached_result is not None:
                chunk_pbar.set_postfix({"状态": "使用缓存"})
                smpl_output = cached_result
            else:
                try:
                    chunk_pbar.set_postfix({"状态": "计算中"})
                    smpl_output = self.estimate_3d_pose(keypoints, frame.shape[:2])
                    with self.cache_lock:
                        self.cache_result(keypoints, smpl_output)
                except Exception as e:
                    chunk_pbar.set_postfix({"状态": f"错误: {str(e)}"})
                    continue
            
            # 保存结果
            with self.file_lock:
                # 保存可视化结果
                self.save_visualization(frame, smpl_output, frame_count)
                
                # 保存SMPL参数
                smpl_params_path = os.path.join(output_dir, f"frame_{frame_count:04d}_smpl_params.xlsx")
                self.save_smpl_params(smpl_output, smpl_params_path)
            
            chunk_pbar.set_postfix({"状态": "完成"})
        
        chunk_pbar.close()
        cap.release()
    
    def process_pose_data(self, pose_data, video_path, output_dir):
        """处理姿态数据，进行3D重建"""
        # 计算每个块的大小
        chunk_size = len(pose_data) // self.num_threads
        if chunk_size == 0:
            chunk_size = 1
        
        # 创建总体进度条
        total_pbar = tqdm(total=len(pose_data), 
                         desc="总体进度", 
                         position=0,
                         leave=True)
        
        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # 提交任务
            futures = []
            for i in range(0, len(pose_data), chunk_size):
                chunk_data = pose_data[i:i + chunk_size]
                future = executor.submit(
                    self.process_chunk,
                    chunk_data,
                    video_path,
                    output_dir,
                    i // chunk_size
                )
                futures.append(future)
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    total_pbar.update(chunk_size)
                except Exception as e:
                    print(f"处理块时发生错误: {str(e)}")
        
        total_pbar.close()
        
        # 创建合并进度条
        merge_pbar = tqdm(total=4, desc="合并结果", position=0)
        
        # 合并结果
        self.merge_results(output_dir, merge_pbar)
        
        merge_pbar.close()
    
    def merge_results(self, output_dir, merge_pbar):
        """合并处理结果"""
        # 收集所有基本参数文件
        basic_files = []
        for file in os.listdir(output_dir):
            if file.endswith('_basic.xlsx'):
                basic_files.append(os.path.join(output_dir, file))
        
        if basic_files:
            merge_pbar.set_description("合并基本参数")
            # 读取并合并所有基本参数
            dfs = []
            for file in basic_files:
                df = pd.read_excel(file)
                dfs.append(df)
            
            # 合并所有数据框
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # 保存合并后的结果
            merged_df.to_excel(os.path.join(output_dir, "merged_basic_params.xlsx"), index=False)
            
            # 删除原始文件
            for file in basic_files:
                os.remove(file)
            merge_pbar.update(1)
        
        # 收集所有顶点文件
        vertex_files = []
        for file in os.listdir(output_dir):
            if file.endswith('_vertices_') and file.endswith('.xlsx'):
                vertex_files.append(os.path.join(output_dir, file))
        
        if vertex_files:
            merge_pbar.set_description("合并顶点数据")
            # 按块号排序
            vertex_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # 读取并合并所有顶点数据
            dfs = []
            for file in vertex_files:
                df = pd.read_excel(file)
                dfs.append(df)
            
            # 合并所有数据框
            merged_df = pd.concat(dfs, axis=1)
            
            # 保存合并后的结果
            merged_df.to_excel(os.path.join(output_dir, "merged_vertices.xlsx"), index=False)
            
            # 删除原始文件
            for file in vertex_files:
                os.remove(file)
            merge_pbar.update(1)
        
        # 收集所有关节文件
        joint_files = []
        for file in os.listdir(output_dir):
            if file.endswith('_joints.xlsx'):
                joint_files.append(os.path.join(output_dir, file))
        
        if joint_files:
            merge_pbar.set_description("合并关节数据")
            # 读取并合并所有关节数据
            dfs = []
            for file in joint_files:
                df = pd.read_excel(file)
                dfs.append(df)
            
            # 合并所有数据框
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # 保存合并后的结果
            merged_df.to_excel(os.path.join(output_dir, "merged_joints.xlsx"), index=False)
            
            # 删除原始文件
            for file in joint_files:
                os.remove(file)
            merge_pbar.update(1)
        
        # 合并所有信息文件
        info_files = []
        for file in os.listdir(output_dir):
            if file.endswith('_info.json'):
                info_files.append(os.path.join(output_dir, file))
        
        if info_files:
            merge_pbar.set_description("合并信息文件")
            # 读取并合并所有信息
            merged_info = {
                'total_frames': 0,
                'vertices_shape': None,
                'joints_shape': None,
                'total_vertices': 0,
                'total_joints': 0,
                'chunk_size': 0,
                'num_vertex_chunks': 0
            }
            
            for file in info_files:
                with open(file, 'r') as f:
                    info = json.load(f)
                    merged_info['total_frames'] += 1
                    if merged_info['vertices_shape'] is None:
                        merged_info['vertices_shape'] = info['vertices_shape']
                    if merged_info['joints_shape'] is None:
                        merged_info['joints_shape'] = info['joints_shape']
                    merged_info['total_vertices'] = max(merged_info['total_vertices'], info['total_vertices'])
                    merged_info['total_joints'] = max(merged_info['total_joints'], info['total_joints'])
                    merged_info['chunk_size'] = max(merged_info['chunk_size'], info['chunk_size'])
                    merged_info['num_vertex_chunks'] += info['num_vertex_chunks']
            
            # 保存合并后的信息
            with open(os.path.join(output_dir, "merged_info.json"), 'w') as f:
                json.dump(merged_info, f, indent=4)
            
            # 删除原始文件
            for file in info_files:
                os.remove(file)
            merge_pbar.update(1)
    
    def estimate_3d_pose(self, keypoints_2d, image_shape):
        """估计3D姿态"""
        print("\n=== 开始3D姿态估计 ===")
        print(f"输入关键点数量: {len(keypoints_2d)}")
        
        # 检查缓存
        cached_result = self.get_cached_result(keypoints_2d)
        if cached_result is not None:
            print("使用缓存的结果")
            return cached_result
        
        # 将关键点转换为SMPL格式
        smpl_keypoints = self.convert_to_smpl_format(keypoints_2d)
        print(f"SMPL关键点形状: {smpl_keypoints.shape}")
        
        # 初始化SMPL参数，设置requires_grad=True
        batch_size = 1
        betas = torch.zeros(batch_size, 10, device=self.device, requires_grad=True)
        expression = torch.zeros(batch_size, 10, device=self.device, requires_grad=True)
        global_orient = torch.zeros(batch_size, 3, device=self.device, requires_grad=True)
        body_pose = torch.zeros(batch_size, 69, device=self.device, requires_grad=True)
        
        # 使用SMPL模型
        try:
            # 确保所有参数都在正确的设备上
            betas = betas.to(self.device)
            expression = expression.to(self.device)
            global_orient = global_orient.to(self.device)
            body_pose = body_pose.to(self.device)
            
            smpl_output = self.model(
                betas=betas,
                expression=expression,
                body_pose=body_pose,
                global_orient=global_orient,
                return_verts=True,
                return_full_pose=True,
                pose2rot=True
            )
            
        except Exception as e:
            print(f"\nSMPL模型计算失败: {str(e)}")
            raise e
        
        # 计算投影损失并优化参数
        loss = self.compute_projection_loss(smpl_output, keypoints_2d, image_shape)
        print(f"\n初始损失值: {loss.item()}")
        
        # 创建优化器
        optimizer = torch.optim.Adam([
            betas, expression, global_orient, body_pose
        ], lr=0.01)
        
        # 优化参数（减少迭代次数）
        for i in range(50):  # 从100次减少到50次
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 重新计算SMPL输出
            smpl_output = self.model(
                betas=betas,
                expression=expression,
                body_pose=body_pose,
                global_orient=global_orient,
                return_verts=True,
                return_full_pose=True,
                pose2rot=True
            )
            
            # 重新计算损失
            loss = self.compute_projection_loss(smpl_output, keypoints_2d, image_shape)
            
            if (i + 1) % 10 == 0:  # 每10次迭代打印一次损失
                print(f"迭代 {i+1}, 损失: {loss.item()}")
        
        print(f"\n最终损失值: {loss.item()}")
        print("=== 3D姿态估计完成 ===\n")
        
        # 缓存结果
        self.cache_result(keypoints_2d, smpl_output)
        
        return smpl_output
    
    def compute_projection_loss(self, smpl_output, keypoints, image_shape):
        """计算2D投影损失"""
        try:
            # 获取SMPL关键点
            smpl_keypoints = smpl_output.joints[:, list(keypoints.keys())]
            print(f"\n投影损失计算:")
            print(f"SMPL关键点形状: {smpl_keypoints.shape}")
            
            # 投影到图像平面
            projected_keypoints = self.project_to_image(smpl_keypoints, image_shape)
            print(f"投影后关键点形状: {projected_keypoints.shape}")
            
            # 计算与检测到的关键点的距离
            loss = 0
            for i, (smpl_idx, (x, y)) in enumerate(keypoints.items()):
                pred_x, pred_y = projected_keypoints[0, i]
                # 创建目标张量，不需要梯度
                target_x = torch.tensor(x, device=self.device, dtype=torch.float32, requires_grad=False)
                target_y = torch.tensor(y, device=self.device, dtype=torch.float32, requires_grad=False)
                loss += F.mse_loss(pred_x, target_x)
                loss += F.mse_loss(pred_y, target_y)
            
            return loss
            
        except Exception as e:
            print(f"\n投影损失计算失败: {str(e)}")
            print("张量形状信息:")
            print(f"smpl_output.joints: {smpl_output.joints.shape}")
            print(f"keypoints: {len(keypoints)}")
            print(f"image_shape: {image_shape}")
            raise e
    
    def project_to_image(self, keypoints_3d, image_shape):
        """将3D关键点投影到图像平面"""
        # 简单的正交投影
        scale = min(image_shape) / 2
        keypoints_2d = keypoints_3d[:, :, :2] * scale + torch.tensor([image_shape[1]/2, image_shape[0]/2], device=self.device)
        return keypoints_2d
    
    def convert_to_smpl_format(self, keypoints_2d):
        """将关键点转换为SMPL格式"""
        try:
            # 创建SMPL关键点数组（SMPL有29个关键点）
            smpl_keypoints = torch.zeros((1, 29, 3), device=self.device)
            
            # 将关键点映射到SMPL关键点
            for smpl_idx, (x, y) in keypoints_2d.items():
                # 将2D坐标转换为3D坐标（这里使用简单的深度估计）
                z = 0.5  # 假设一个固定的深度值
                smpl_keypoints[0, smpl_idx] = torch.tensor([x, y, z], device=self.device)
            
            print(f"\n关键点转换:")
            print(f"输入关键点数量: {len(keypoints_2d)}")
            print(f"输出SMPL关键点形状: {smpl_keypoints.shape}")
            
            return smpl_keypoints
            
        except Exception as e:
            print(f"\n关键点转换失败: {str(e)}")
            print("输入数据:")
            print(f"keypoints_2d: {keypoints_2d}")
            raise e
    
    def save_visualization(self, original_frame, smpl_output, frame_count):
        """保存可视化结果"""
        # 获取SMPL网格
        vertices = smpl_output.vertices[0].detach().cpu().numpy()
        faces = self.model.faces
        
        # 创建简单的2D投影可视化
        plt.figure(figsize=(12, 8))
        
        # 绘制原始图像
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        
        # 绘制3D投影
        plt.subplot(122)
        # 简单的正交投影
        vertices_2d = vertices[:, :2]
        vertices_2d = vertices_2d - np.min(vertices_2d)
        vertices_2d = vertices_2d / np.max(vertices_2d)
        vertices_2d = vertices_2d * [original_frame.shape[1], original_frame.shape[0]]
        
        # 绘制网格
        for face in faces:
            plt.plot(vertices_2d[face, 0], vertices_2d[face, 1], 'b-', alpha=0.3)
        
        plt.title('3D重建投影')
        plt.axis('off')
        
        # 保存图像
        plt.savefig(f"output/visualization/frame_{frame_count:04d}.png")
        plt.close()
    
    def save_smpl_params(self, smpl_output, output_path):
        """保存SMPL参数到多个Excel文件"""
        try:
            # 提取参数
            params = {
                'betas': smpl_output.betas.detach().cpu().numpy(),
                'global_orient': smpl_output.global_orient.detach().cpu().numpy(),
                'body_pose': smpl_output.body_pose.detach().cpu().numpy(),
                'vertices': smpl_output.vertices.detach().cpu().numpy(),
                'joints': smpl_output.joints.detach().cpu().numpy()
            }
            
            # 创建基础文件名（不包含扩展名）
            base_path = output_path.replace('.xlsx', '')
            
            # 分别保存不同类型的参数
            # 1. 保存基本参数（betas, global_orient, body_pose）
            basic_params = {}
            for param_name in ['betas', 'global_orient', 'body_pose']:
                param_value = params[param_name]
                if param_value.ndim == 2:
                    for i in range(param_value.shape[1]):
                        basic_params[f'{param_name}_{i}'] = param_value[:, i]
                else:
                    basic_params[param_name] = param_value.flatten()
            
            # 保存基本参数
            basic_df = pd.DataFrame(basic_params)
            basic_df.to_excel(f"{base_path}_basic.xlsx", index=False)
            
            # 2. 保存vertices（分块保存）
            vertices = params['vertices']
            vertices_flat = vertices.reshape(vertices.shape[0], -1)
            chunk_size = 1000  # 每块1000个顶点
            
            for i in range(0, vertices_flat.shape[1], chunk_size):
                chunk_end = min(i + chunk_size, vertices_flat.shape[1])
                chunk_df = pd.DataFrame({
                    f'vertex_{j}': vertices_flat[:, j] 
                    for j in range(i, chunk_end)
                })
                chunk_df.to_excel(f"{base_path}_vertices_{i//chunk_size}.xlsx", index=False)
            
            # 3. 保存joints
            joints = params['joints']
            joints_flat = joints.reshape(joints.shape[0], -1)
            joints_df = pd.DataFrame({
                f'joint_{i}': joints_flat[:, i] 
                for i in range(joints_flat.shape[1])
            })
            joints_df.to_excel(f"{base_path}_joints.xlsx", index=False)
            
            # 保存参数信息
            info = {
                'vertices_shape': vertices.shape,
                'joints_shape': joints.shape,
                'total_vertices': vertices.shape[1],
                'total_joints': joints.shape[1],
                'chunk_size': chunk_size,
                'num_vertex_chunks': (vertices_flat.shape[1] + chunk_size - 1) // chunk_size
            }
            
            with open(f"{base_path}_info.json", 'w') as f:
                json.dump(info, f, indent=4)
            
        except Exception as e:
            print(f"保存SMPL参数失败: {str(e)}")
            print("参数形状信息:")
            for param_name, param_value in params.items():
                print(f"{param_name}: {param_value.shape}")
            raise e

    def reconstruct_poses(self, input_file: str, output_file: str) -> None:
        """从2D关键点重建3D姿态"""
        try:
            logger.info(f"开始3D重建: {input_file}")
            
            # 读取2D关键点数据
            with h5py.File(input_file, 'r') as f:
                keypoints_2d = f['keypoints'][:]
                confidence = f['confidence'][:]
                frame_count = f.attrs['frame_count']
                fps = f.attrs['fps']
                video_name = f.attrs['video_name']
            
            # 转换数据格式
            keypoints_3d = self.convert_2d_to_3d_format(keypoints_2d, confidence)
            
            # 初始化结果存储
            results = {
                'vertices': [],
                'joints': [],
                'global_orient': [],
                'body_pose': [],
                'betas': [],
                'transl': []
            }
            
            # 处理每一帧
            for i in tqdm(range(frame_count), desc="3D重建"):
                # 获取当前帧的关键点
                frame_keypoints = torch.tensor(keypoints_3d[i:i+1], 
                                            dtype=torch.float32, 
                                            device=self.device)
                
                # 使用SMPL模型重建
                smpl_output = self.model(
                    global_orient=torch.zeros(1, 3, device=self.device),
                    body_pose=torch.zeros(1, 69, device=self.device),
                    betas=torch.zeros(1, 10, device=self.device),
                    transl=torch.zeros(1, 3, device=self.device)
                )
                
                # 更新结果
                results['vertices'].append(smpl_output.vertices[0].cpu().numpy())
                results['joints'].append(smpl_output.joints[0].cpu().numpy())
                results['global_orient'].append(smpl_output.global_orient[0].cpu().numpy())
                results['body_pose'].append(smpl_output.body_pose[0].cpu().numpy())
                results['betas'].append(smpl_output.betas[0].cpu().numpy())
                results['transl'].append(smpl_output.transl[0].cpu().numpy())
            
            # 保存结果
            self.save_results(results, output_file, frame_count, fps, video_name)
            
            logger.info(f"3D重建完成: {output_file}")
            
        except Exception as e:
            logger.error(f"3D重建失败: {str(e)}")
            raise
            
    def convert_2d_to_3d_format(self, keypoints_2d: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        """将2D关键点转换为3D重建所需的格式"""
        # 确保关键点格式正确
        keypoints_3d = np.zeros((len(keypoints_2d), 17, 3))
        keypoints_3d[:, :, :2] = keypoints_2d
        keypoints_3d[:, :, 2] = confidence
        return keypoints_3d

    def save_results(self, results: Dict, output_file: str, frame_count: int, fps: float, video_name: str):
        """保存重建结果到HDF5文件"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # 保存顶点数据
            f.create_dataset('vertices', data=results['vertices'])
            
            # 保存关节点数据
            f.create_dataset('joints', data=results['joints'])
            
            # 保存SMPL参数
            params = f.create_group('parameters')
            for key, value in results['parameters'].items():
                params.create_dataset(key, data=value)
                
            # 保存元数据
            metadata = f.create_group('metadata')
            metadata.attrs['total_frames'] = frame_count
            metadata.attrs['fps'] = fps
            metadata.attrs['video_name'] = video_name
            
        logger.info(f"结果已保存到: {output_file}")
        
    def load_results(self, input_file: str) -> Dict:
        """从HDF5文件加载重建结果"""
        results = {}
        
        with h5py.File(input_file, 'r') as f:
            # 加载顶点数据
            results['vertices'] = f['vertices'][:]
            
            # 加载关节点数据
            results['joints'] = f['joints'][:]
            
            # 加载SMPL参数
            results['parameters'] = {}
            for key in f['parameters'].keys():
                results['parameters'][key] = f['parameters'][key][:]
                
            # 加载元数据
            results['metadata'] = dict(f['metadata'].attrs)
            
        return results

def main():
    # 创建3D重建器实例
    reconstructor = Pose3DReconstructor()
    
    # 处理检测结果
    detection_dir = "./output/detection"
    reconstruction_dir = "./output/reconstruction"
    os.makedirs(reconstruction_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(detection_dir):
        for file in files:
            if file.endswith('_pose_data.csv'):
                # 读取姿态数据
                pose_data = pd.read_csv(os.path.join(root, file))
                
                # 检查数据格式
                print(f"\n检查CSV文件: {file}")
                print(f"列名: {pose_data.columns.tolist()}")
                print(f"数据行数: {len(pose_data)}")
                
                # 转换为字典列表
                pose_data = pose_data.to_dict('records')
                
                # 获取对应的视频文件
                video_name = file.replace('_pose_data.csv', '')
                video_path = os.path.join("./videos", video_name + ".mp4")
                
                print(f"\n处理视频: {video_path}")
                reconstructor.process_pose_data(pose_data, video_path, reconstruction_dir)

if __name__ == "__main__":
    main()