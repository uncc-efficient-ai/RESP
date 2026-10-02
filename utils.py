import importlib.machinery
import importlib.util
import sys
import os
import hashlib
from pathlib import Path


def load_module(package_path):
    package = Path(package_path).resolve()
    module_name = '_resp_plugin_' + hashlib.sha256(str(package).encode()).hexdigest()[:16]
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, package / '__init__.py', submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


import logging, traceback


def setup_logger(config, multi_process=False):
    if not multi_process:
        # 创建一个 logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # 设置为最低级别，确保捕获所有日志
        # 创建一个文件处理器，用于写入日志文件
        file_handler = logging.FileHandler(config.logger.log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
    else:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"logging to rank: {rank}")
        log_file = config.logger.log_file_path.replace('.txt', f'_rank{rank}.txt')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
    # 创建一个控制台处理器，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个格式器，定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
