import os
import numpy as np
import torch
import h5py
import pybullet as p
import smplx
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import json
import time
from pathlib import Path
import pybullet_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SMPLVisualizer:
    """SMPL模型可视化类"""
    def __init__(self):
        self.debug_lines = []
        self.debug_texts = []
        self.debug_arrows = []
        
    def clear_visualization(self):
        """清除所有可视化元素"""
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        for text_id in self.debug_texts:
            p.removeUserDebugItem(text_id)
        for arrow_id in self.debug_arrows:
            p.removeUserDebugItem(arrow_id)
        self.debug_lines = []
        self.debug_texts = []
        self.debug_arrows = []
        
    def draw_skeleton(self, joint_positions: np.ndarray):
        """绘制骨骼结构"""
        # 定义骨骼连接关系
        connections = [
            (0, 1), (1, 4), (4, 7),  # 左腿
            (0, 2), (2, 5), (5, 8),  # 右腿
            (0, 3), (3, 9), (9, 12), # 左臂
            (3, 10), (10, 13),       # 右臂
            (3, 11), (11, 14)        # 头部
        ]
        
        # 绘制骨骼连线
        for start, end in connections:
            line_id = p.addUserDebugLine(
                lineFromXYZ=joint_positions[start],
                lineToXYZ=joint_positions[end],
                lineColorRGB=[0, 1, 0],  # 绿色
                lineWidth=2,
                lifeTime=0
            )
            self.debug_lines.append(line_id)
            
    def draw_joint_angles(self, body_id: int, position: np.ndarray):
        """绘制关节角度"""
        # 获取关节位置和方向
        pos, orn = p.getBasePositionAndOrientation(body_id)
        euler_angles = p.getEulerFromQuaternion(orn)
        
        # 显示欧拉角
        text_id = p.addUserDebugText(
            text=f"Euler: {np.round(euler_angles, 2)}",
            textPosition=position + [0, 0.1, 0],
            textColorRGB=[1, 1, 0],
            textSize=1.2,
            lifeTime=0
        )
        self.debug_texts.append(text_id)
        
        # 绘制旋转轴箭头
        arrow_length = 0.1
        for i, angle in enumerate(euler_angles):
            axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB对应XYZ轴
            axis_directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            
            arrow_id = p.addUserDebugArrow(
                fromPos=position,
                toPos=position + np.array(axis_directions[i]) * arrow_length,
                lineColorRGB=axis_colors[i],
                lineWidth=2,
                lifeTime=0
            )
            self.debug_arrows.append(arrow_id)
            
    def update_visualization(self, joint_positions: np.ndarray, body_ids: List[int]):
        """更新可视化"""
        self.clear_visualization()
        self.draw_skeleton(joint_positions)
        
        # 更新每个关节的可视化
        for i, body_id in enumerate(body_ids):
            self.draw_joint_angles(body_id, joint_positions[i])

class SMPLPhysicsModel:
    def __init__(self, device: Optional[torch.device] = None):
        """初始化SMPL物理模型"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"使用设备: {self.device}")
        
        # 初始化PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 加载地面
        self.ground = p.loadURDF("plane.urdf")
        
        # 加载SMPL模型
        self.smpl = self.load_smpl_model()
        
        # 初始化可视化器
        self.visualizer = SMPLVisualizer()
        
    def load_smpl_model(self):
        """加载SMPL模型到PyBullet"""
        # 这里需要根据实际的SMPL模型文件路径进行调整
        model_path = "models/smpl/SMPL_NEUTRAL.pkl"
        
        # 创建SMPL模型
        from smplx import SMPL
        model = SMPL(
            model_path=model_path,
            gender='neutral',
            batch_size=1,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_transl=True
        ).to(self.device)
        
        return model
        
    def process_data(self, data_path: str) -> Dict:
        """处理数据"""
        try:
            logging.info(f"开始处理数据: {data_path}")
            
            # 读取数据
            with h5py.File(data_path, 'r') as f:
                # 读取基本参数
                global_orient = torch.tensor(f['global_orient'][:], device=self.device)
                body_pose = torch.tensor(f['body_pose'][:], device=self.device)
                betas = torch.tensor(f['betas'][:], device=self.device)
                transl = torch.tensor(f['transl'][:], device=self.device)
                
                # 读取顶点和关节点
                vertices = torch.tensor(f['vertices'][:], device=self.device)
                joints = torch.tensor(f['joints'][:], device=self.device)
                
                # 读取元数据
                frame_count = f.attrs['frame_count']
                fps = f.attrs['fps']
                video_name = f.attrs['video_name']
            
            # 转换数据格式
            physics_params = self.convert_smpl_to_physics_format({
                'global_orient': global_orient,
                'body_pose': body_pose,
                'betas': betas,
                'transl': transl
            })
            
            # 初始化物理环境
            p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81)
            
            # 创建地面
            ground_shape = p.createCollisionShape(p.GEOM_PLANE)
            ground = p.createMultiBody(0, ground_shape)
            
            # 创建人体模型
            human_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
            human = p.createMultiBody(1, human_shape)
            
            # 处理每一帧
            results = {
                'vertices': [],
                'joints': [],
                'global_orient': [],
                'body_pose': [],
                'betas': [],
                'transl': [],
                'physics_params': []
            }
            
            for i in tqdm(range(frame_count), desc="处理物理模型"):
                # 更新人体模型位置
                p.resetBasePositionAndOrientation(
                    human,
                    [physics_params['transl'][i, 0], physics_params['transl'][i, 1], physics_params['transl'][i, 2]],
                    p.getQuaternionFromEuler([0, 0, 0])
                )
                
                # 应用物理模拟
                for _ in range(10):  # 模拟10步
                    p.stepSimulation()
                
                # 获取更新后的位置
                pos, _ = p.getBasePositionAndOrientation(human)
                
                # 更新结果
                results['vertices'].append(vertices[i].cpu().numpy())
                results['joints'].append(joints[i].cpu().numpy())
                results['global_orient'].append(physics_params['global_orient'][i].cpu().numpy())
                results['body_pose'].append(physics_params['body_pose'][i].cpu().numpy())
                results['betas'].append(physics_params['betas'][i].cpu().numpy())
                results['transl'].append(physics_params['transl'][i].cpu().numpy())
                results['physics_params'].append({
                    'position': pos,
                    'velocity': p.getBaseVelocity(human)[0],
                    'angular_velocity': p.getBaseVelocity(human)[1]
                })
            
            # 断开物理连接
            p.disconnect()
            
            # 转换为numpy数组
            for key in ['vertices', 'joints', 'global_orient', 'body_pose', 'betas', 'transl']:
                results[key] = np.array(results[key])
            
            logging.info("数据处理完成")
            return results
            
        except Exception as e:
            logging.error(f"处理失败: {str(e)}")
            raise
        
    def convert_smpl_to_physics_format(self, smpl_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将SMPL参数转换为物理模拟所需的格式"""
        physics_params = {
            'global_orient': smpl_params['global_orient'],
            'body_pose': smpl_params['body_pose'],
            'betas': smpl_params['betas'],
            'transl': smpl_params['transl']
        }
        return physics_params
        
    def visualize_data(self, vertices: np.ndarray, joints: np.ndarray, 
                      parameters: Dict[str, np.ndarray], output_dir: Path):
        """可视化SMPL数据"""
        try:
            # 创建输出目录
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 设置matplotlib
            plt.style.use('seaborn')
            
            # 创建3D图形
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制顶点
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      c='blue', alpha=0.1, label='Vertices')
            
            # 绘制关节点
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
                      c='red', s=100, label='Joints')
            
            # 绘制骨架连接
            self._draw_skeleton(ax, joints)
            
            # 设置图形属性
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('SMPL Model Visualization')
            ax.legend()
            
            # 保存图形
            plt.savefig(output_dir / 'smpl_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 创建关节角度可视化
            self._visualize_joint_angles(parameters, output_dir)
            
            logger.info(f"可视化结果已保存到: {output_dir}")
            
        except Exception as e:
            logger.error(f"可视化数据时出错: {str(e)}")
            raise e
            
    def _draw_skeleton(self, ax, joints):
        """绘制骨架连接"""
        # SMPL骨架连接定义
        connections = [
            # 躯干
            (0, 1), (1, 2), (2, 3), (3, 4),  # 脊椎
            (1, 5), (2, 6),  # 左右髋
            (5, 7), (6, 8),  # 左右膝
            (7, 9), (8, 10),  # 左右踝
            
            # 手臂
            (3, 11), (3, 12),  # 左右肩
            (11, 13), (12, 14),  # 左右肘
            (13, 15), (14, 16),  # 左右腕
            
            # 头部
            (4, 17), (17, 18), (18, 19)  # 头部
        ]
        
        # 绘制连接线
        for start, end in connections:
            ax.plot([joints[start, 0], joints[end, 0]],
                   [joints[start, 1], joints[end, 1]],
                   [joints[start, 2], joints[end, 2]],
                   'g-', alpha=0.5)
            
    def _visualize_joint_angles(self, parameters: Dict[str, np.ndarray], output_dir: Path):
        """可视化关节角度"""
        try:
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Joint Angles Visualization')
            
            # 绘制全局方向
            axes[0, 0].plot(parameters['global_orient'])
            axes[0, 0].set_title('Global Orientation')
            axes[0, 0].set_xlabel('Frame')
            axes[0, 0].set_ylabel('Angle (rad)')
            
            # 绘制身体姿态
            axes[0, 1].plot(parameters['body_pose'])
            axes[0, 1].set_title('Body Pose')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('Angle (rad)')
            
            # 绘制形状参数
            axes[1, 0].plot(parameters['betas'])
            axes[1, 0].set_title('Shape Parameters')
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('Value')
            
            # 绘制平移
            axes[1, 1].plot(parameters['transl'])
            axes[1, 1].set_title('Translation')
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('Distance')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图形
            plt.savefig(output_dir / 'joint_angles.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"可视化关节角度时出错: {str(e)}")
            raise e
        
    def close(self):
        """关闭物理引擎"""
        p.disconnect(self.physics_client)

def main():
    """主函数"""
    # 设置路径
    smpl_model_path = "models/smpl_model.pt"
    input_file = "output/reconstruction/merged_basic_params.h5"
    output_file = "output/reconstruction/physics_model.h5"
    
    try:
        # 创建物理模型实例（启用GUI）
        physics_model = SMPLPhysicsModel(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 处理HDF5文件
        physics_model.process_data(input_file)
        
        logger.info("物理模型处理完成")
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
    finally:
        # 关闭物理引擎
        if 'physics_model' in locals():
            physics_model.close()

if __name__ == "__main__":
    main() 