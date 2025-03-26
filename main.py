import os
import torch
import logging
import argparse
from pathlib import Path
from typing import Optional, List
import h5py
from tqdm import tqdm
from google.colab import drive
from google.colab import files

# 导入各个模块
from pose_detection import PoseDetector
from pose_3d_reconstruction import Pose3DReconstructor
from parallel_merge_results import ParallelResultMerger
from smpl_physics_model import SMPLPhysicsModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_cuda_availability():
    """检查CUDA是否可用并返回合适的设备"""
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，将使用CPU")
        return torch.device('cpu')
    
    try:
        # 测试CUDA是否真正可用
        test_tensor = torch.zeros(1).cuda()
        logger.info(f"CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    except Exception as e:
        logger.error(f"CUDA初始化失败: {str(e)}")
        logger.warning("将使用CPU作为备选")
        return torch.device('cpu')

def mount_google_drive():
    """挂载Google Drive"""
    try:
        drive.mount('/content/drive')
        logger.info("Google Drive已挂载")
    except Exception as e:
        logger.error(f"Google Drive挂载失败: {str(e)}")
        raise

class SMPLPipeline:
    def __init__(self, device: Optional[torch.device] = None):
        """初始化SMPL处理流水线"""
        try:
            if device is None:
                self.device = check_cuda_availability()
            else:
                self.device = device
                
            logger.info(f"使用设备: {self.device}")
            
            # 确保CUDA设备已正确初始化
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 初始化各个模块
            self.pose_detector = PoseDetector(device=self.device)
            self.pose_reconstructor = Pose3DReconstructor(device=self.device)
            self.result_merger = ParallelResultMerger(device=self.device)
            self.physics_model = SMPLPhysicsModel(device=self.device)
            
        except Exception as e:
            logger.error(f"初始化SMPL流水线失败: {str(e)}")
            raise
        
    def process_video(self, video_path: str, output_dir: str) -> None:
        """处理单个视频文件"""
        try:
            video_name = Path(video_path).stem
            logger.info(f"开始处理视频: {video_name}")
            
            # 创建输出目录
            output_path = Path(output_dir) / video_name
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 1. 姿态检测
            pose_output = output_path / "pose_detection.h5"
            self.pose_detector.detect_video(video_path, pose_output)
            
            # 2. 3D重建
            reconstruction_output = output_path / "reconstruction.h5"
            self.pose_reconstructor.reconstruct_poses(pose_output, reconstruction_output)
            
            # 3. 数据合并
            merge_output = output_path / "merged_results.h5"
            self.result_merger.merge_all_results(reconstruction_output, merge_output)
            
            # 4. 物理模型处理
            physics_output = output_path / "physics_results.h5"
            physics_results = self.physics_model.process_data(merge_output)
            
            # 保存物理模型结果
            with h5py.File(physics_output, 'w') as f:
                for key, value in physics_results.items():
                    if key != 'physics_params':
                        f.create_dataset(key, data=value)
                    else:
                        # 特殊处理物理参数
                        physics_group = f.create_group('physics_params')
                        for i, params in enumerate(value):
                            frame_group = physics_group.create_group(f'frame_{i}')
                            frame_group.create_dataset('position', data=params['position'])
                            frame_group.create_dataset('velocity', data=params['velocity'])
                            frame_group.create_dataset('angular_velocity', data=params['angular_velocity'])
            
            logger.info(f"视频处理完成: {video_name}")
            
        except Exception as e:
            logger.error(f"处理视频失败: {str(e)}")
            raise
            
    def process_directory(self, input_dir: Path, output_dir: Path):
        """处理整个目录的视频文件"""
        try:
            # 获取所有视频文件
            video_files = list(input_dir.glob("*.mp4"))
            if not video_files:
                logger.warning(f"在目录 {input_dir} 中没有找到视频文件")
                return
                
            logger.info(f"找到 {len(video_files)} 个视频文件")
            
            # 处理每个视频
            for video_path in tqdm(video_files, desc="处理视频"):
                try:
                    self.process_video(str(video_path), str(output_dir))
                except Exception as e:
                    logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"处理目录时出错: {str(e)}")
            raise e
            
    def visualize_results(self, merged_data_path: str, output_dir: Path):
        """可视化合并后的数据"""
        try:
            # 创建可视化输出目录
            vis_output = output_dir / "visualization"
            vis_output.mkdir(parents=True, exist_ok=True)
            
            # 加载合并后的数据
            with h5py.File(merged_data_path, 'r') as f:
                vertices = f['vertices'][:]
                joints = f['joints'][:]
                parameters = {
                    'global_orient': f['parameters/global_orient'][:],
                    'body_pose': f['parameters/body_pose'][:],
                    'betas': f['parameters/betas'][:],
                    'transl': f['parameters/transl'][:]
                }
                metadata = dict(f['metadata'].attrs)
            
            # 使用物理模型进行可视化
            self.physics_model.visualize_data(
                vertices=vertices,
                joints=joints,
                parameters=parameters,
                output_dir=vis_output
            )
            
            logger.info(f"可视化结果已保存到: {vis_output}")
            
        except Exception as e:
            logger.error(f"可视化数据时出错: {str(e)}")
            raise e

def main():
    parser = argparse.ArgumentParser(description="SMPL处理流水线")
    parser.add_argument("--input", type=str, required=True, help="Google Drive中的输入视频文件夹路径")
    parser.add_argument("--output", type=str, required=True, help="Google Drive中的输出目录路径")
    parser.add_argument("--device", type=str, default=None, help="指定设备 (cuda/cpu)")
    parser.add_argument("--visualize", action="store_true", help="是否进行可视化")
    
    args = parser.parse_args()
    
    try:
        # 挂载Google Drive
        mount_google_drive()
        
        # 设置设备
        if args.device:
            try:
                device = torch.device(args.device)
                if device.type == 'cuda' and not torch.cuda.is_available():
                    logger.warning("指定的CUDA设备不可用，将使用CPU")
                    device = torch.device('cpu')
            except Exception as e:
                logger.error(f"设备设置失败: {str(e)}")
                device = torch.device('cpu')
        else:
            device = check_cuda_availability()
            
        # 转换路径为Google Drive路径
        input_path = Path('/content/drive/MyDrive') / args.input
        output_path = Path('/content/drive/MyDrive') / args.output
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化流水线
        pipeline = SMPLPipeline(device=device)
        
        # 处理输入目录中的所有视频
        video_files = list(input_path.glob("*.mp4"))
        if not video_files:
            logger.warning(f"在目录 {input_path} 中没有找到视频文件")
            return 1
            
        logger.info(f"找到 {len(video_files)} 个视频文件")
        
        # 处理每个视频
        for video_path in tqdm(video_files, desc="处理视频"):
            try:
                pipeline.process_video(str(video_path), str(output_path))
                # 在每个视频处理完后清理GPU内存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
                continue
                
        # 如果启用了可视化，处理合并后的数据
        if args.visualize:
            merged_data_path = output_path / "merged_data.h5"
            if merged_data_path.exists():
                pipeline.visualize_results(str(merged_data_path), output_path)
            else:
                logger.warning("未找到合并后的数据文件，跳过可视化")
                
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}") 