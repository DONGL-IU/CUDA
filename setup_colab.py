import os
import sys
import subprocess
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def activate_venv():
    """激活虚拟环境"""
    try:
        venv_python = Path("venv/bin/python")
        if not venv_python.exists():
            raise FileNotFoundError("虚拟环境Python解释器未找到")
            
        # 将虚拟环境的Python添加到系统路径
        os.environ["VIRTUAL_ENV"] = str(venv_python.parent.parent)
        os.environ["PATH"] = f"{venv_python.parent}:{os.environ['PATH']}"
        
        # 完全替换Python环境
        sys.executable = str(venv_python)
        sys.prefix = str(venv_python.parent.parent)
        sys.base_prefix = str(venv_python.parent.parent)
        sys.base_exec_prefix = str(venv_python.parent.parent)
        
        # 添加虚拟环境的site-packages到Python路径
        site_packages = venv_python.parent.parent / "lib" / "python3.10" / "site-packages"
        if site_packages.exists():
            sys.path = [str(site_packages)] + [p for p in sys.path if "python3.11" not in p]
            
        # 验证Python版本
        version_check = subprocess.run([str(venv_python), "--version"], capture_output=True, text=True)
        if "Python 3.10" not in version_check.stdout:
            raise RuntimeError(f"虚拟环境Python版本不正确: {version_check.stdout}")
            
        # 验证sys.version
        if "3.10" not in sys.version:
            raise RuntimeError(f"sys.version不正确: {sys.version}")
            
        logger.info(f"虚拟环境激活成功: {version_check.stdout.strip()}")
        return True
    except Exception as e:
        logger.error(f"激活虚拟环境失败: {str(e)}")
        return False

def setup_colab_environment():
    """配置Colab环境"""
    try:
        # 检查是否在Colab环境中
        try:
            import google.colab
            logger.info("检测到Colab环境")
        except ImportError:
            logger.error("未检测到Colab环境，请确保在Google Colab中运行此脚本")
            return False
            
        # 安装Python 3.10
        logger.info("正在安装Python 3.10...")
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "python3.10", "python3.10-dev", "python3.10-venv"], check=True)
        
        # 创建虚拟环境
        logger.info("创建Python 3.10虚拟环境...")
        subprocess.run(["python3.10", "-m", "venv", "venv"], check=True)
        
        # 激活虚拟环境
        if not activate_venv():
            return False
        
        # 升级pip
        logger.info("升级pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # 安装依赖包
        logger.info("安装依赖包...")
        packages = [
            "torch==2.0.1",
            "numpy==1.23.5",
            "opencv-python==4.7.0.72",
            "pandas==1.5.3",
            "h5py==3.8.0",
            "tqdm==4.65.0",
            "matplotlib==3.7.1",
            "ultralytics==8.0.0",
            "smplx==0.1.28",
            "scipy==1.10.1"
        ]
        for package in packages:
            logger.info(f"安装 {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            
        # 创建必要的目录
        logger.info("创建项目目录...")
        directories = [
            "models/smpl",
            "models/yolo",
            "output/visualization",
            "output/opensim",
            "output/detection",
            "output/reconstruction"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        # 下载SMPL模型
        logger.info("下载SMPL模型...")
        smpl_url = "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_NEUTRAL.pkl"
        smpl_path = Path("models/smpl/SMPL_NEUTRAL.pkl")
        if not smpl_path.exists():
            subprocess.run(["wget", "-O", str(smpl_path), smpl_url], check=True)
            
        logger.info("环境配置完成！")
        return True
        
    except Exception as e:
        logger.error(f"环境配置失败: {str(e)}")
        return False

if __name__ == "__main__":
    if setup_colab_environment():
        logger.info("环境配置成功，虚拟环境已激活")
        # 打印当前Python环境信息
        logger.info(f"当前Python解释器: {sys.executable}")
        logger.info(f"Python版本: {sys.version}")
        
        # 验证Python版本
        version_check = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        logger.info(f"Python版本验证: {version_check.stdout.strip()}")
        
        # 验证sys.path
        logger.info("Python路径:")
        for path in sys.path:
            logger.info(f"  {path}")
    else:
        logger.error("环境配置失败") 