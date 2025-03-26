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
    """激活虚拟环境（增强版）"""
    try:
        venv_dir = Path("venv").absolute()
        venv_python = venv_dir / "bin" / "python"
        
        if not venv_python.exists():
            raise FileNotFoundError(f"虚拟环境Python未找到: {venv_python}")

        # 彻底清理PATH中的其他Python路径
        os.environ["PATH"] = f"{venv_dir/'bin'}:{os.environ.get('PATH', '')}"
        os.environ["PATH"] = ":".join(
            p for p in os.environ["PATH"].split(":") 
            if "python" not in p.lower() or str(venv_dir) in p
        )
        
        # 完全替换系统路径
        os.environ["VIRTUAL_ENV"] = str(venv_dir)
        sys.executable = str(venv_python)
        sys.prefix = str(venv_dir)
        sys.base_prefix = str(venv_dir)
        sys.base_exec_prefix = str(venv_dir)
        
        # 重载sys.path
        lib_path = venv_dir / "lib" / "python3.10"
        sys.path = [
            str(lib_path / "site-packages"),
            str(lib_path),
            *[p for p in sys.path if "python3.11" not in p]
        ]
        
        # 验证版本
        version = subprocess.run(
            [str(venv_python), "-c", "import sys; print(sys.version)"],
            capture_output=True, text=True
        )
        if "3.10" not in version.stdout:
            raise RuntimeError(f"版本验证失败: {version.stdout}")
            
        logger.info(f"虚拟环境激活成功\nPython路径: {venv_python}\n版本: {version.stdout}")
        return True
        
    except Exception as e:
        logger.error(f"激活失败: {type(e).__name__}: {str(e)}")
        return False

def setup_colab_environment():
    """配置Colab环境（完整修正版）"""
    try:
        # 检查是否在Colab环境中
        try:
            import google.colab
            logger.info("检测到Colab环境")
        except ImportError:
            logger.error("未检测到Colab环境，请确保在Google Colab中运行此脚本")
            return False
            
        # 1. 安装Python 3.10
        logger.info("安装Python 3.10...")
        subprocess.run([
            "apt-get", "update"
        ], check=True)
        subprocess.run([
            "apt-get", "install", "-y", 
            "python3.10", 
            "python3.10-dev", 
            "python3.10-distutils",
            "python3.10-venv"
        ], check=True)
        
        # 2. 创建干净的虚拟环境
        logger.info("创建虚拟环境...")
        venv_cmd = [
            "/usr/bin/python3.10",  # 明确指定Python 3.10路径
            "-m", "venv", 
            "--clear",  # 清除现有环境
            "--copies",  # 使用独立副本而非符号链接
            "--system-site-packages=False",  # 禁止继承系统包
            "venv"
        ]
        subprocess.run(venv_cmd, check=True)
        
        # 3. 激活环境
        if not activate_venv():
            raise RuntimeError("虚拟环境激活失败")
            
        # 4. 升级pip
        logger.info("升级pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # 5. 安装依赖包
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
            
        # 6. 创建必要的目录
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
            
        # 7. 下载SMPL模型
        logger.info("下载SMPL模型...")
        smpl_url = "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_NEUTRAL.pkl"
        smpl_path = Path("models/smpl/SMPL_NEUTRAL.pkl")
        if not smpl_path.exists():
            subprocess.run(["wget", "-O", str(smpl_path), smpl_url], check=True)
            
        logger.info("环境配置完成！")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e.cmd}\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"配置失败: {type(e).__name__}: {str(e)}")
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