import logging
import os

class Logger:
    """日志工具类：同时输出到控制台和日志文件"""

    def __init__(self, log_path):
        # 创建日志目录
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 初始化logger
        self.logger = logging.getLogger("PMAT-SASRec-Logger")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()  # 清空重复处理器

        # 格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 文件处理器
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, msg):
        """记录info级别日志"""
        self.logger.info(msg)

    def warning(self, msg):
        """记录warning级别日志"""
        self.logger.warning(msg)

    def error(self, msg):
        """记录error级别日志"""
        self.logger.error(msg)