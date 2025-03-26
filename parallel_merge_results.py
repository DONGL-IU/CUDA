import os
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
from typing import List, Dict, Any, Optional, Tuple
import logging
import sys
import psutil
import time
import h5py
import numpy as np
from multiprocessing import cpu_count
import platform
import torch
from pathlib import Path
import concurrent.futures
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

class ParallelResultMerger:
    def __init__(self, device: Optional[torch.device] = None):
        """初始化并行结果合并器"""
        try:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = device
                
            logger.info(f"使用设备: {self.device}")
            
            # 设置线程数
            self.num_threads = min(8, os.cpu_count() or 1)
            logger.info(f"使用线程数: {self.num_threads}")
            
        except Exception as e:
            logger.error(f"初始化并行结果合并器失败: {str(e)}")
            raise
    
    def merge_all_results(self, input_file: str, output_file: str) -> None:
        """合并所有处理结果"""
        try:
            logger.info(f"开始合并结果: {input_file}")
            
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
            
            # 创建输出目录
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建进度条
            pbar = tqdm(total=4, desc="合并结果")
            
            # 1. 合并顶点数据
            pbar.set_description("合并顶点数据")
            merged_vertices = self.merge_vertices(vertices)
            pbar.update(1)
            
            # 2. 合并关节点数据
            pbar.set_description("合并关节点数据")
            merged_joints = self.merge_joints(joints)
            pbar.update(1)
            
            # 3. 合并SMPL参数
            pbar.set_description("合并SMPL参数")
            merged_parameters = self.merge_parameters(parameters)
            pbar.update(1)
            
            # 4. 保存合并结果
            pbar.set_description("保存合并结果")
            self.save_merged_results(
                output_path,
                merged_vertices,
                merged_joints,
                merged_parameters,
                metadata
            )
            pbar.update(1)
            
            pbar.close()
            logger.info(f"结果合并完成: {output_path}")
            
        except Exception as e:
            logger.error(f"合并结果失败: {str(e)}")
            raise
    
    def merge_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """合并顶点数据"""
        try:
            # 确保顶点数据是3D的
            if vertices.ndim == 2:
                vertices = vertices.reshape(-1, vertices.shape[1] // 3, 3)
            
            # 合并所有帧的顶点
            merged_vertices = np.concatenate(vertices, axis=0)
            return merged_vertices
            
        except Exception as e:
            logger.error(f"合并顶点数据失败: {str(e)}")
            raise
    
    def merge_joints(self, joints: np.ndarray) -> np.ndarray:
        """合并关节点数据"""
        try:
            # 确保关节点数据是3D的
            if joints.ndim == 2:
                joints = joints.reshape(-1, joints.shape[1] // 3, 3)
            
            # 合并所有帧的关节点
            merged_joints = np.concatenate(joints, axis=0)
            return merged_joints
            
        except Exception as e:
            logger.error(f"合并关节点数据失败: {str(e)}")
            raise
    
    def merge_parameters(self, parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """合并SMPL参数"""
        try:
            merged_params = {}
            for key, value in parameters.items():
                # 确保参数是3D的
                if value.ndim == 2:
                    value = value.reshape(-1, value.shape[1] // 3, 3)
                merged_params[key] = np.concatenate(value, axis=0)
            return merged_params
            
        except Exception as e:
            logger.error(f"合并SMPL参数失败: {str(e)}")
            raise
    
    def save_merged_results(
        self,
        output_path: Path,
        vertices: np.ndarray,
        joints: np.ndarray,
        parameters: Dict[str, np.ndarray],
        metadata: Dict
    ) -> None:
        """保存合并后的结果"""
        try:
            with h5py.File(output_path, 'w') as f:
                # 保存顶点数据
                f.create_dataset('vertices', data=vertices)
                
                # 保存关节点数据
                f.create_dataset('joints', data=joints)
                
                # 保存SMPL参数
                params_group = f.create_group('parameters')
                for key, value in parameters.items():
                    params_group.create_dataset(key, data=value)
                
                # 保存元数据
                for key, value in metadata.items():
                    f.attrs[key] = value
                    
            logger.info(f"合并结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存合并结果失败: {str(e)}")
            raise

def main():
    # 创建并行结果合并器实例
    merger = ParallelResultMerger()
    
    # 处理重建结果
    reconstruction_dir = Path("./output/reconstruction")
    output_dir = Path("./output/merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理所有重建结果文件
    for result_file in reconstruction_dir.glob("*.h5"):
        output_file = output_dir / f"merged_{result_file.name}"
        merger.merge_all_results(str(result_file), str(output_file))

if __name__ == "__main__":
    main() 