import os
import pandas as pd
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
from typing import List, Dict, Any, Optional, Tuple, Union, TypeVar, Sequence
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
    def __init__(self, num_threads: Optional[int] = None):
        """初始化结果合并器"""
        try:
            self.num_threads = num_threads or os.cpu_count() or 4
            logger.info(f"使用 {self.num_threads} 个线程进行并行处理")
            
        except Exception as e:
            logger.error(f"初始化结果合并器失败: {str(e)}")
            raise
    
    def merge_results(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """合并多个结果文件"""
        try:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            logger.info(f"开始合并结果: {input_dir}")
            
            # 创建输出目录
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取所有结果文件
            result_files = list(input_dir.glob("*_3d.h5"))
            if not result_files:
                logger.warning(f"未找到结果文件: {input_dir}")
                return
            
            logger.info(f"找到 {len(result_files)} 个结果文件")
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # 提交所有任务
                futures = []
                for result_file in result_files:
                    future = executor.submit(
                        self.merge_single_result,
                        result_file,
                        output_dir
                    )
                    futures.append(future)
                
                # 等待所有任务完成
                for future in tqdm(as_completed(futures), total=len(futures), desc="合并结果"):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"合并结果失败: {str(e)}")
            
            logger.info(f"结果合并完成: {output_dir}")
            
        except Exception as e:
            logger.error(f"合并结果失败: {str(e)}")
            raise
    
    def merge_single_result(self, result_file: Path, output_dir: Path) -> None:
        """合并单个结果文件"""
        try:
            # 读取结果文件
            with h5py.File(result_file, 'r') as f:
                poses_3d = f['poses_3d'][:]
                betas = f['betas'][:]
                frame_count = f.attrs['frame_count']
                fps = f.attrs['fps']
                width = f.attrs['width']
                height = f.attrs['height']
            
            # 创建输出文件
            output_file = output_dir / f"merged_{result_file.stem}.h5"
            
            # 保存合并后的结果
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('poses_3d', data=poses_3d)
                f.create_dataset('betas', data=betas)
                f.attrs['frame_count'] = frame_count
                f.attrs['fps'] = fps
                f.attrs['width'] = width
                f.attrs['height'] = height
                f.attrs['source_file'] = str(result_file)
            
            logger.info(f"已合并结果: {output_file}")
            
        except Exception as e:
            logger.error(f"合并单个结果失败: {str(e)}")
            raise

def main():
    """主函数"""
    try:
        # 创建结果合并器实例
        merger = ParallelResultMerger()
        
        # 设置输入和输出目录
        input_dir = Path("./output/reconstruction")
        output_dir = Path("./output/merged")
        
        # 合并结果
        merger.merge_results(input_dir, output_dir)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 