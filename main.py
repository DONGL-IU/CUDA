import os
import torch
import logging
import argparse
from pathlib import Path
from typing import Optional, List
import h5py
from tqdm import tqdm
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入各个模块
try:
    from pose_detection import PoseDetector
    from pose_3d_reconstruction import Pose3DReconstructor
    from parallel_merge_results import ParallelResultMerger
    logger.info("成功导入所有模块")
except Exception as e:
    logger.error(f"模块导入失败: {str(e)}")
    raise

def is_colab_environment():
    """检查是否在Google Colab环境中运行"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def download_from_drive(drive_path: str, local_path: str):
    """从Google Drive下载文件到本地"""
    try:
        from google.colab import files
        from google.colab import drive
        
        # 挂载Google Drive
        logger.info("正在挂载Google Drive...")
        drive.mount('/content/drive')
        logger.info("Google Drive挂载成功")
        
        # 构建完整的Drive路径
        full_drive_path = Path('/content/drive/MyDrive') / drive_path
        logger.info(f"尝试访问Drive路径: {full_drive_path}")
        
        # 检查路径是否存在
        if not full_drive_path.exists():
            logger.error(f"Drive路径不存在: {full_drive_path}")
            return False
            
        # 确保目标目录存在
        local_path = Path(local_path)
        logger.info(f"创建本地目录: {local_path}")
        local_path.mkdir(parents=True, exist_ok=True)
        
        # 如果是目录，复制所有文件
        if full_drive_path.is_dir():
            logger.info(f"正在复制目录: {full_drive_path} -> {local_path}")
            # 获取所有视频文件
            video_files = list(full_drive_path.glob("*.mp4"))
            if not video_files:
                logger.warning(f"在目录 {full_drive_path} 中没有找到视频文件")
                return False
                
            logger.info(f"找到 {len(video_files)} 个视频文件")
            
            # 复制每个视频文件
            for video_file in video_files:
                try:
                    target_file = local_path / video_file.name
                    logger.info(f"正在复制文件: {video_file} -> {target_file}")
                    shutil.copy2(video_file, target_file)
                except Exception as e:
                    logger.error(f"复制文件 {video_file} 失败: {str(e)}")
                    continue
                    
            return True
        else:
            logger.error(f"指定的路径不是目录: {full_drive_path}")
            return False
            
    except Exception as e:
        logger.error(f"从Google Drive下载失败: {str(e)}")
        return False

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
            try:
                self.pose_detector = PoseDetector(device=self.device)
                self.pose_reconstructor = Pose3DReconstructor(device=self.device)
                self.result_merger = ParallelResultMerger(device=self.device)
            except Exception as e:
                logger.error(f"模块初始化失败: {str(e)}")
                raise
            
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

def main():
    parser = argparse.ArgumentParser(description="SMPL处理流水线")
    parser.add_argument("--input", type=str, required=True, help="输入视频文件夹路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录路径")
    parser.add_argument("--device", type=str, default=None, help="指定设备 (cuda/cpu)")
    parser.add_argument("--no-drive", action="store_true", help="不使用Google Drive")
    
    args = parser.parse_args()
    
    try:
        # 检查环境
        is_colab = is_colab_environment()
        logger.info(f"运行环境: {'Google Colab' if is_colab else '本地环境'}")
        
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
            
        # 处理输入路径
        if is_colab and not args.no_drive:
            # 在Colab中，将文件下载到本地
            local_input = "/content/input_videos"
            logger.info(f"准备从Google Drive下载文件: {args.input} -> {local_input}")
            if download_from_drive(args.input, local_input):
                input_path = Path(local_input)
                logger.info(f"成功设置输入路径: {input_path}")
            else:
                logger.error("无法从Google Drive下载输入文件")
                return 1
        else:
            input_path = Path(args.input)
            
        # 设置输出路径
        if is_colab and not args.no_drive:
            output_path = Path("/content/output_results")
        else:
            output_path = Path(args.output)
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录已创建: {output_path}")
        
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
                
        # 如果是在Colab中运行，将结果上传回Google Drive
        if is_colab and not args.no_drive:
            try:
                from google.colab import files
                logger.info("准备上传结果到Google Drive...")
                files.download(str(output_path))
                logger.info("结果上传成功")
            except Exception as e:
                logger.error(f"上传结果到Google Drive失败: {str(e)}")
                
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    # 打印环境信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
    print(f"运行环境: {'Google Colab' if is_colab_environment() else '本地环境'}")
    
    exit(main()) 