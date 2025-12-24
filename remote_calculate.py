"""
Remote Calculation Script for NTKCAP
远程计算独立脚本

用法:
    python remote_calculate.py <path1> <path2> ... [options]

示例:
    python remote_calculate.py "D:\NTKCAP\Patient_data\patient1\2024_11_13" --fast
    python remote_calculate.py "D:\NTKCAP\Patient_data\patient1\2024_11_13" "D:\NTKCAP\Patient_data\patient2\2024_11_13" --fast --no-gait
    python remote_calculate.py --config config.json

说明:
    此脚本可在远程服务器上独立运行，不依赖GUI
    需要先激活 conda 环境: conda activate ntkcap_fast
"""

import sys
import os
import json
import argparse
from datetime import datetime
from multiprocessing import Queue
import warnings

# ============== 环境变量设置 ==============
# 抑制警告信息
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["NVBX_VERBOSE"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:torch.distributed.optim.zero_redundancy_optimizer"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# 抑制 NumPy 和 PyTorch 警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 设置 CUDA 和 TensorRT 环境变量（如果存在）
def setup_cuda_tensorrt_env(pwd):
    """设置 CUDA 和 TensorRT 环境变量 (Cross-platform: Linux/Windows)"""
    import platform
    is_windows = platform.system() == "Windows"
    path_sep = ";" if is_windows else ":"
    
    # CUDA 路径检测 - Platform-specific paths
    if is_windows:
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        ]
    else:
        # Linux paths
        cuda_paths = [
            "/usr/local/cuda-11.8",
            "/usr/local/cuda",
            "/opt/cuda",
        ]
    
    cuda_found = False
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            os.environ["CUDA_HOME"] = cuda_path
            os.environ["CUDA_PATH"] = cuda_path  # For compatibility
            cuda_bin = os.path.join(cuda_path, "bin")
            cuda_lib = os.path.join(cuda_path, "lib64" if not is_windows else "lib")
            
            # Update PATH for binaries
            current_path = os.environ.get("PATH", "")
            if cuda_bin not in current_path:
                os.environ["PATH"] = f"{cuda_bin}{path_sep}{current_path}"
            
            # On Linux, update LD_LIBRARY_PATH for libraries
            if not is_windows:
                current_ld = os.environ.get("LD_LIBRARY_PATH", "")
                if cuda_lib not in current_ld:
                    os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}{path_sep}{current_ld}" if current_ld else cuda_lib
            
            print(f"✅ CUDA 路径设置: {cuda_path}")
            cuda_found = True
            break
    
    if not cuda_found:
        print("⚠️  未检测到 CUDA 安装")
    
    # TensorRT 路径设置
    tensorrt_dir = os.path.join(pwd, "NTK_CAP", "ThirdParty", "TensorRT-8.6.1.6")
    if os.path.exists(tensorrt_dir):
        tensorrt_lib = os.path.join(tensorrt_dir, "lib")
        tensorrt_bin = os.path.join(tensorrt_dir, "bin") if is_windows else None
        
        os.environ["TENSORRT_ROOT"] = tensorrt_dir
        os.environ["TENSORRT_DIR"] = tensorrt_dir
        os.environ["TRT_LIBPATH"] = tensorrt_lib
        
        if is_windows:
            # Windows: Add to PATH
            current_path = os.environ.get("PATH", "")
            if tensorrt_lib not in current_path:
                os.environ["PATH"] = f"{tensorrt_lib}{path_sep}{current_path}"
            if tensorrt_bin and tensorrt_bin not in current_path:
                os.environ["PATH"] = f"{tensorrt_bin}{path_sep}{os.environ['PATH']}"
        else:
            # Linux: Add to LD_LIBRARY_PATH
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if tensorrt_lib not in current_ld:
                os.environ["LD_LIBRARY_PATH"] = f"{tensorrt_lib}{path_sep}{current_ld}" if current_ld else tensorrt_lib
        
        print(f"✅ TensorRT 路径设置: {tensorrt_dir}")
    else:
        print("⚠️  未检测到 TensorRT 安装")
    
    # MMDeploy SDK (libmmdeploy_trt_net.so) - Linux only
    if not is_windows:
        mmdeploy_lib = os.path.join(pwd, "NTK_CAP", "ThirdParty", "mmdeploy", "build", "lib")
        if os.path.exists(mmdeploy_lib):
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if mmdeploy_lib not in current_ld:
                os.environ["LD_LIBRARY_PATH"] = f"{mmdeploy_lib}{path_sep}{current_ld}" if current_ld else mmdeploy_lib
            print(f"✅ MMDeploy SDK 路径设置: {mmdeploy_lib}")

# Add NTK_CAP script_py directory to Python path
script_py_path = os.path.join(os.path.dirname(__file__), 'NTK_CAP', 'script_py')
if script_py_path not in sys.path:
    sys.path.insert(0, script_py_path)

from NTK_CAP.script_py.NTK_Cap import mp_marker_calculate


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='NTKCAP Remote Calculation Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单个路径计算
  python remote_calculate.py "D:\\NTKCAP\\Patient_data\\patient1\\2024_11_13"
  
  # 多个路径计算
  python remote_calculate.py "path1" "path2" "path3"
  
  # 快速计算模式
  python remote_calculate.py "path1" --fast
  
  # 不计算步态
  python remote_calculate.py "path1" --no-gait
  
  # 使用配置文件
  python remote_calculate.py --config calculation_config.json
        """
    )
    
    parser.add_argument(
        'paths',
        nargs='*',
        help='计算路径列表 (Patient_data/patient_id/date 格式)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='启用快速计算模式'
    )
    
    parser.add_argument(
        '--no-gait',
        action='store_true',
        help='不计算步态分析'
    )
    
    parser.add_argument(
        '--pwd',
        default=None,
        help='工作目录路径 (默认: 脚本所在目录)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='使用配置文件 (JSON格式)'
    )
    
    parser.add_argument(
        '--task-filter',
        type=str,
        help='任务过滤配置文件 (JSON格式)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出模式'
    )
    
    return parser.parse_args()


def load_config_file(config_path):
    """从配置文件加载计算参数"""
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def validate_paths(pwd, paths):
    """验证路径是否存在"""
    invalid_paths = []
    
    for path in paths:
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(path):
            path = os.path.join(pwd, path)
        
        if not os.path.exists(path):
            invalid_paths.append(path)
    
    if invalid_paths:
        print("❌ 以下路径不存在:")
        for p in invalid_paths:
            print(f"  - {p}")
        return False
    
    return True


def print_calculation_summary(pwd, paths, fast_cal, gait, task_filter_dict):
    """打印计算摘要"""
    print("\n" + "="*60)
    print("NTKCAP 远程计算任务")
    print("="*60)
    print(f"工作目录: {pwd}")
    print(f"计算路径数量: {len(paths)}")
    print(f"快速计算: {'是' if fast_cal else '否'}")
    print(f"步态分析: {'是' if gait else '否'}")
    print(f"任务过滤: {'是' if task_filter_dict else '否'}")
    print("\n计算路径列表:")
    for i, path in enumerate(paths, 1):
        print(f"  {i}. {path}")
        if task_filter_dict and path in task_filter_dict:
            tasks = task_filter_dict[path]
            if tasks:
                print(f"     过滤任务: {', '.join(tasks)}")
    print("="*60 + "\n")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 确定工作目录
    if args.pwd:
        pwd = os.path.abspath(args.pwd)
    else:
        pwd = os.path.dirname(os.path.abspath(__file__))
    
    # 设置 CUDA 和 TensorRT 环境变量
    setup_cuda_tensorrt_env(pwd)
    
    # 从配置文件加载或使用命令行参数
    if args.config:
        config = load_config_file(args.config)
        paths = config.get('paths', [])
        fast_cal = config.get('fast_cal', False)
        gait = config.get('gait', True)
        task_filter_dict = config.get('task_filter_dict', None)
    else:
        if not args.paths:
            print("❌ 错误: 请提供计算路径或配置文件")
            print("使用 --help 查看帮助")
            sys.exit(1)
        
        paths = args.paths
        fast_cal = args.fast
        gait = not args.no_gait
        
        # 加载任务过滤配置
        task_filter_dict = None
        if args.task_filter:
            task_filter_dict = load_config_file(args.task_filter)
    
    # 验证路径
    if not validate_paths(pwd, paths):
        sys.exit(1)
    
    # 打印计算摘要
    if args.verbose:
        print_calculation_summary(pwd, paths, fast_cal, gait, task_filter_dict)
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"⏱️  计算开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 执行计算 (不使用进度队列，因为是远程无GUI环境)
        print("\n🚀 开始计算...")
        mp_marker_calculate(
            PWD=pwd,
            calculate_path_list=paths,
            fast_cal=fast_cal,
            gait=gait,
            progress_queue=None,  # 远程模式不需要GUI进度反馈
            task_filter_dict=task_filter_dict
        )
        
        # 记录结束时间
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        print("\n" + "="*60)
        print("✅ 计算完成!")
        print("="*60)
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {elapsed_time}")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        print("\n" + "="*60)
        print("❌ 计算失败!")
        print("="*60)
        print(f"错误信息: {str(e)}")
        print(f"失败时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"运行时长: {elapsed_time}")
        print("="*60 + "\n")
        
        if args.verbose:
            import traceback
            print("详细错误追踪:")
            traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
