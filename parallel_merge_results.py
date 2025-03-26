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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParallelResultMerger:
    def __init__(self, output_dir: str, device: Optional[torch.device] = None):
        """初始化并行结果合并器"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"使用设备: {self.device}")
        
        # 设置输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置线程数
        self.max_workers = min(16, os.cpu_count() or 8)
        logger.info(f"使用 {self.max_workers} 个工作线程")
        
        # 设置内存限制
        self.memory_limit = int(psutil.virtual_memory().total * 0.3)  # 使用30%的系统内存
        logger.info(f"内存限制: {self.memory_limit / (1024**3):.2f} GB")
        
        # 初始化进度条
        self.merge_pbar = None
        self.file_pbar = None
        self.pbar_lock = threading.Lock()
        
    def merge_all_results(self):
        """合并所有结果"""
        logger.info("开始合并所有结果...")
        
        # 创建总体进度条
        self.merge_pbar = tqdm(total=4, desc="总体进度", position=0)
        
        try:
            # 1. 合并基本参数
            self.merge_basic_params()
            self.merge_pbar.update(1)
            
            # 2. 合并顶点数据
            self.merge_vertices()
            self.merge_pbar.update(1)
            
            # 3. 合并关节点数据
            self.merge_joints()
            self.merge_pbar.update(1)
            
            # 4. 合并信息文件
            self.merge_info()
            self.merge_pbar.update(1)
            
        finally:
            # 清理进度条
            if self.merge_pbar:
                self.merge_pbar.close()
            if self.file_pbar:
                self.file_pbar.close()
                
        logger.info("合并完成")
        
    def merge_basic_params(self):
        """合并基本参数"""
        logger.info("开始合并基本参数...")
        
        # 查找所有基本参数文件
        param_files = list(self.output_dir.glob("frame_*_smpl_params.h5"))
        if not param_files:
            logger.warning("未找到基本参数文件")
            return
            
        # 创建文件处理进度条
        self.file_pbar = tqdm(total=len(param_files), desc="处理文件", position=1)
        
        # 使用线程池处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file in param_files:
                future = executor.submit(self.process_file_chunk, file)
                futures.append(future)
                
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"处理文件时出错: {str(e)}")
                    
        # 合并所有结果
        self.merge_hdf5_files(param_files, self.output_dir / "merged_basic_params.h5")
        
    def merge_vertices(self):
        """合并顶点数据"""
        logger.info("开始合并顶点数据...")
        
        # 查找所有顶点数据文件
        vertex_files = list(self.output_dir.glob("frame_*_smpl_params_vertices_*.h5"))
        if not vertex_files:
            logger.warning("未找到顶点数据文件")
            return
            
        # 创建文件处理进度条
        self.file_pbar = tqdm(total=len(vertex_files), desc="处理文件", position=1)
        
        # 使用线程池处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file in vertex_files:
                future = executor.submit(self.process_file_chunk, file)
                futures.append(future)
                
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"处理文件时出错: {str(e)}")
                    
        # 合并所有结果
        self.merge_hdf5_files(vertex_files, self.output_dir / "merged_vertices.h5")
        
    def merge_joints(self):
        """合并关节点数据"""
        logger.info("开始合并关节点数据...")
        
        # 查找所有关节点数据文件
        joint_files = list(self.output_dir.glob("frame_*_joints.h5"))
        if not joint_files:
            logger.warning("未找到关节点数据文件")
            return
            
        # 创建文件处理进度条
        self.file_pbar = tqdm(total=len(joint_files), desc="处理文件", position=1)
        
        # 使用线程池处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for file in joint_files:
                future = executor.submit(self.process_file_chunk, file)
                futures.append(future)
                
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"处理文件时出错: {str(e)}")
                    
        # 合并所有结果
        self.merge_hdf5_files(joint_files, self.output_dir / "merged_joints.h5")
        
    def merge_info(self):
        """合并信息文件"""
        logger.info("开始合并信息文件...")
        
        # 查找信息文件
        info_file = self.output_dir / "reconstruction_summary.json"
        if not info_file.exists():
            logger.warning("未找到信息文件")
            return
            
        # 读取并合并信息
        with open(info_file, 'r') as f:
            info_data = json.load(f)
            
        # 保存合并后的信息
        output_file = self.output_dir / "merged_info.json"
        with open(output_file, 'w') as f:
            json.dump(info_data, f, indent=4)
            
        logger.info(f"信息已保存到: {output_file}")
        
    def process_file_chunk(self, file_path: Path):
        """处理单个文件块"""
        try:
            # 更新进度条描述
            with self.pbar_lock:
                self.file_pbar.set_description(f"处理: {file_path.name}")
                
            # 读取HDF5文件
            with h5py.File(file_path, 'r') as f:
                # 获取数据集信息
                datasets = list(f.keys())
                
                # 检查内存使用
                if psutil.Process().memory_info().rss > self.memory_limit:
                    logger.warning("内存使用超过限制，等待释放...")
                    while psutil.Process().memory_info().rss > self.memory_limit:
                        time.sleep(1)
                        
            # 更新进度
            with self.pbar_lock:
                self.file_pbar.update(1)
                
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
            raise e
            
    def merge_hdf5_files(self, input_files: List[Path], output_file: Path):
        """合并多个HDF5文件"""
        logger.info(f"开始合并 {len(input_files)} 个文件到 {output_file}")
        
        # 创建输出文件
        with h5py.File(output_file, 'w') as out_f:
            # 获取第一个文件的结构
            with h5py.File(input_files[0], 'r') as f:
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        # 创建数据集
                        shape = list(f[key].shape)
                        shape[0] = sum(h5py.File(fname, 'r')[key].shape[0] for fname in input_files)
                        out_f.create_dataset(key, shape=shape, dtype=f[key].dtype)
                        
                        # 复制属性
                        for attr_name, attr_value in f[key].attrs.items():
                            out_f[key].attrs[attr_name] = attr_value
                            
            # 合并数据
            offset = 0
            for input_file in input_files:
                with h5py.File(input_file, 'r') as f:
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            # 获取数据
                            data = f[key][:]
                            
                            # 写入数据
                            out_f[key][offset:offset + len(data)] = data
                            
                offset += len(data)
                
        logger.info(f"合并完成: {output_file}")
        
    def log_memory_usage(self):
        """记录内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"内存使用: {memory_info.rss / (1024**3):.2f} GB")

def main():
    """主函数"""
    # 设置输出目录
    reconstruction_dir = "./output/reconstruction"
    
    # 检查目录是否存在
    if not os.path.exists(reconstruction_dir):
        print(f"错误: 目录 {reconstruction_dir} 不存在")
        return
    
    # 创建并行合并器实例（自动检测最优线程数）
    merger = ParallelResultMerger(reconstruction_dir)
    
    try:
        # 合并所有结果
        merger.merge_all_results()
    except Exception as e:
        logger.error(f"合并过程中发生错误: {str(e)}")
        return

if __name__ == "__main__":
    main() 