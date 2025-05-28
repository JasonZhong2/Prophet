import os
import subprocess
import sys
import logging
import traceback
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置镜像源
MIRROR_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

def create_directories():
    """创建必要的目录"""
    try:
        directories = ['dependencies', 'dependencies/wheels']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"创建目录: {directory}")
            else:
                logger.info(f"目录已存在: {directory}")
    except Exception as e:
        logger.error(f"创建目录时出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def check_pip():
    """检查pip是否可用"""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', '--version'],
            capture_output=True,
            text=True
        )
        logger.info(f"Pip版本信息: {result.stdout.strip()}")
        return True
    except Exception as e:
        logger.error(f"检查pip时出错: {str(e)}")
        return False

def download_with_retry(max_retries=3, retry_delay=5):
    """带重试机制的下载函数"""
    for attempt in range(max_retries):
        try:
            logger.info(f"尝试下载依赖包 (第 {attempt + 1} 次)")
            
            # 下载所有依赖包到wheels目录
            result = subprocess.run([
                sys.executable, 
                '-m', 
                'pip', 
                'download', 
                '-r', 
                'requirements.txt',
                '-d', 
                'dependencies/wheels',
                '--verbose',
                '--no-cache-dir',
                '-i', MIRROR_URL,
                '--timeout', '100',  # 增加超时时间到100秒
                '--retries', '3'     # 添加重试次数
            ], capture_output=True, text=True)
            
            # 输出命令执行结果
            logger.info("命令输出:")
            logger.info(result.stdout)
            
            if result.stderr:
                logger.error("错误输出:")
                logger.error(result.stderr)
            
            # 检查wheels目录是否为空
            wheels_dir = 'dependencies/wheels'
            if os.path.exists(wheels_dir):
                files = os.listdir(wheels_dir)
                if files:
                    logger.info(f"成功下载 {len(files)} 个文件到 {wheels_dir}")
                    logger.info("下载的文件列表:")
                    for file in files:
                        logger.info(f"- {file}")
                    return True
                else:
                    logger.error(f"{wheels_dir} 目录为空！")
            else:
                logger.error(f"{wheels_dir} 目录不存在！")
                
        except Exception as e:
            logger.error(f"下载过程中出现错误: {str(e)}")
            logger.error(traceback.format_exc())
            
            if attempt < max_retries - 1:
                logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                logger.error("达到最大重试次数，下载失败")
                return False
    
    return False

def download_dependencies():
    """下载所有依赖包到本地"""
    logger.info("开始下载依赖包...")
    
    try:
        # 创建目录
        create_directories()
        
        # 检查requirements.txt是否存在
        if not os.path.exists('requirements.txt'):
            logger.error("requirements.txt 文件不存在！")
            return
        
        # 检查Python和pip版本
        python_version = sys.version
        logger.info(f"Python版本: {python_version}")
        
        # 检查pip是否可用
        if not check_pip():
            logger.error("pip不可用，请检查pip安装")
            return
        
        # 读取requirements.txt内容
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        logger.info(f"Requirements.txt内容:\n{requirements}")
        
        # 使用重试机制下载依赖
        if download_with_retry():
            logger.info("依赖包下载完成！")
        else:
            logger.error("依赖包下载失败！")
            
    except Exception as e:
        logger.error(f"下载过程中出现错误: {str(e)}")
        logger.error(traceback.format_exc())
        return

def install_from_local():
    """从本地安装依赖包"""
    logger.info("开始从本地安装依赖包...")
    
    try:
        # 检查wheels目录是否存在且不为空
        wheels_dir = 'dependencies/wheels'
        if not os.path.exists(wheels_dir):
            logger.error(f"{wheels_dir} 目录不存在！")
            return
        
        files = os.listdir(wheels_dir)
        if not files:
            logger.error(f"{wheels_dir} 目录为空！")
            return
        
        # 从本地wheels目录安装所有包
        logger.info("开始执行pip install命令...")
        result = subprocess.run([
            sys.executable,
            '-m',
            'pip',
            'install',
            '--no-index',
            '--find-links=dependencies/wheels',
            '-r',
            'requirements.txt',
            '--verbose',
            '--no-cache-dir'
        ], capture_output=True, text=True)
        
        # 输出命令执行结果
        logger.info("命令输出:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.error("错误输出:")
            logger.error(result.stderr)
            
    except Exception as e:
        logger.error(f"安装过程中出现错误: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
    logger.info("依赖包安装完成！")

if __name__ == '__main__':
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--install':
            install_from_local()
        else:
            download_dependencies()
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        logger.error(traceback.format_exc()) 