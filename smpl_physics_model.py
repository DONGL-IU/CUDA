import os
import torch
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import pybullet as p
import pybullet_data
import inspect

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

class SMPLPhysicsModel:
    def __init__(self, device: Optional[torch.device] = None):
        """初始化SMPL物理模型"""
        try:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device
                
            logger.info(f"使用设备: {self.device}")
            
            # 初始化PyBullet
            self.init_pybullet()
            
            # 加载SMPL模型
            self.load_smpl_model()
            
        except Exception as e:
            logger.error(f"初始化SMPL物理模型失败: {str(e)}")
            raise
    
    def init_pybullet(self):
        """初始化PyBullet环境"""
        try:
            # 连接到PyBullet服务器
            p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # 设置重力
            p.setGravity(0, 0, -9.81)
            
            # 加载地面
            p.loadURDF("plane.urdf")
            
            logger.info("PyBullet环境初始化成功")
            
        except Exception as e:
            logger.error(f"PyBullet环境初始化失败: {str(e)}")
            raise
    
    def load_smpl_model(self):
        """加载SMPL模型"""
        try:
            # 加载SMPL模型文件
            model_path = 'models/smpl/SMPL_NEUTRAL.pkl'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"SMPL模型文件不存在: {model_path}")
            
            # 创建SMPL模型
            self.model = p.loadURDF(
                "models/smpl/smpl.urdf",
                useFixedBase=True,
                globalScaling=1.0
            )
            
            logger.info("SMPL模型加载成功")
            
        except Exception as e:
            logger.error(f"SMPL模型加载失败: {str(e)}")
            raise
    
    def process_data(self, input_file: str) -> Dict:
        """处理SMPL数据并应用物理模型"""
        try:
            logger.info(f"开始处理数据: {input_file}")
            
            # 读取输入数据
            with h5py.File(input_file, 'r') as f:
                vertices = f['vertices'][:]
                joints = f['joints'][:]
                parameters = {
                    'global_orient': f['parameters/global_orient'][:],
                    'body_pose': f['parameters/body_pose'][:],
                    'betas': f['parameters/betas'][:],
                    'transl': f['parameters/transl'][:]
                }
                metadata = dict(f.attrs)
            
            # 创建进度条
            pbar = tqdm(total=len(vertices), desc="应用物理模型")
            
            # 初始化结果存储
            physics_results = {
                'vertices': [],
                'joints': [],
                'physics_params': []
            }
            
            # 处理每一帧
            for i in range(len(vertices)):
                # 更新SMPL模型状态
                self.update_model_state(
                    vertices[i],
                    joints[i],
                    parameters['global_orient'][i],
                    parameters['body_pose'][i],
                    parameters['betas'][i],
                    parameters['transl'][i]
                )
                
                # 应用物理模拟
                physics_params = self.apply_physics()
                
                # 获取更新后的状态
                updated_vertices, updated_joints = self.get_model_state()
                
                # 保存结果
                physics_results['vertices'].append(updated_vertices)
                physics_results['joints'].append(updated_joints)
                physics_results['physics_params'].append(physics_params)
                
                pbar.update(1)
            
            pbar.close()
            logger.info("物理模型处理完成")
            
            return physics_results
            
        except Exception as e:
            logger.error(f"处理数据失败: {str(e)}")
            raise
    
    def update_model_state(
        self,
        vertices: np.ndarray,
        joints: np.ndarray,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
        betas: np.ndarray,
        transl: np.ndarray
    ):
        """更新SMPL模型状态"""
        try:
            # 设置全局方向
            p.resetBasePositionAndOrientation(
                self.model,
                transl,
                p.getQuaternionFromEuler(global_orient)
            )
            
            # 设置身体姿态
            for i in range(len(body_pose)):
                p.resetJointState(
                    self.model,
                    i,
                    body_pose[i]
                )
            
            # 设置形状参数
            for i in range(len(betas)):
                p.resetJointState(
                    self.model,
                    i + len(body_pose),
                    betas[i]
                )
            
        except Exception as e:
            logger.error(f"更新模型状态失败: {str(e)}")
            raise
    
    def apply_physics(self) -> Dict:
        """应用物理模拟"""
        try:
            # 模拟一步物理
            p.stepSimulation()
            
            # 获取当前状态
            position, orientation = p.getBasePositionAndOrientation(self.model)
            linear_velocity, angular_velocity = p.getBaseVelocity(self.model)
            
            return {
                'position': position,
                'orientation': orientation,
                'linear_velocity': linear_velocity,
                'angular_velocity': angular_velocity
            }
            
        except Exception as e:
            logger.error(f"应用物理模拟失败: {str(e)}")
            raise
    
    def get_model_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取模型当前状态"""
        try:
            # 获取顶点位置
            vertices = np.zeros((6890, 3))
            for i in range(6890):
                vertices[i] = p.getLinkState(self.model, i)[0]
            
            # 获取关节点位置
            joints = np.zeros((24, 3))
            for i in range(24):
                joints[i] = p.getLinkState(self.model, i)[0]
            
            return vertices, joints
            
        except Exception as e:
            logger.error(f"获取模型状态失败: {str(e)}")
            raise
    
    def visualize_data(
        self,
        vertices: np.ndarray,
        joints: np.ndarray,
        parameters: Dict[str, np.ndarray],
        output_dir: Path
    ) -> None:
        """可视化处理结果"""
        try:
            logger.info("开始可视化...")
            
            # 创建输出目录
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建进度条
            pbar = tqdm(total=len(vertices), desc="生成可视化")
            
            # 处理每一帧
            for i in range(len(vertices)):
                # 更新模型状态
                self.update_model_state(
                    vertices[i],
                    joints[i],
                    parameters['global_orient'][i],
                    parameters['body_pose'][i],
                    parameters['betas'][i],
                    parameters['transl'][i]
                )
                
                # 应用物理模拟
                self.apply_physics()
                
                # 获取相机图像
                width = 640
                height = 480
                view_matrix = p.computeViewMatrix(
                    cameraEyePosition=[2, 2, 2],
                    cameraTargetPosition=[0, 0, 0],
                    cameraUpVector=[0, 0, 1]
                )
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=width/height,
                    nearVal=0.1,
                    farVal=100.0
                )
                
                _, _, rgb, _, _ = p.getCameraImage(
                    width,
                    height,
                    view_matrix,
                    proj_matrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL
                )
                
                # 保存图像
                import cv2
                rgb_array = np.array(rgb)
                rgb_array = rgb_array[:, :, :3]
                cv2.imwrite(
                    str(output_dir / f"frame_{i:04d}.png"),
                    cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                )
                
                pbar.update(1)
            
            pbar.close()
            logger.info(f"可视化结果已保存到: {output_dir}")
            
        except Exception as e:
            logger.error(f"可视化失败: {str(e)}")
            raise
    
    def __del__(self):
        """清理PyBullet环境"""
        try:
            p.disconnect()
        except:
            pass

def main():
    # 创建SMPL物理模型实例
    physics_model = SMPLPhysicsModel()
    
    # 处理合并后的结果
    merged_dir = Path("./output/merged")
    output_dir = Path("./output/physics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理所有合并结果文件
    for result_file in merged_dir.glob("merged_*.h5"):
        # 处理数据
        physics_results = physics_model.process_data(str(result_file))
        
        # 保存结果
        output_file = output_dir / f"physics_{result_file.name}"
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('vertices', data=physics_results['vertices'])
            f.create_dataset('joints', data=physics_results['joints'])
            
            # 保存物理参数
            physics_group = f.create_group('physics_params')
            for i, params in enumerate(physics_results['physics_params']):
                frame_group = physics_group.create_group(f'frame_{i}')
                frame_group.create_dataset('position', data=params['position'])
                frame_group.create_dataset('orientation', data=params['orientation'])
                frame_group.create_dataset('linear_velocity', data=params['linear_velocity'])
                frame_group.create_dataset('angular_velocity', data=params['angular_velocity'])

if __name__ == "__main__":
    main() 