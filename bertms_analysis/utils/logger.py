"""
日志管理模块
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "bertms_analysis",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器

    Parameters
    ----------
    name : str, optional
        日志记录器名称，默认为 "bertms_analysis"
    level : str, optional
        日志级别，可选值: DEBUG, INFO, WARNING, ERROR, CRITICAL
        默认为 "INFO"
    log_file : str, optional
        日志文件路径，如果提供则同时输出到文件和控制台
        默认为 None（仅输出到控制台）
    format_string : str, optional
        日志格式字符串，默认为标准格式

    Returns
    -------
    logging.Logger
        配置好的日志记录器

    Examples
    --------
    >>> logger = setup_logger("my_analysis", level="DEBUG")
    >>> logger.info("Analysis started")
    >>> logger.debug("Debug information")
    >>> logger.warning("Warning message")
    """
    # 创建日志记录器
    logger = logging.getLogger(name)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 设置日志级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))

    # 设置日志格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 添加文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 创建默认日志记录器
default_logger = setup_logger()


class LoggerAdapter:
    """
    日志记录器适配器，提供简化的日志记录接口

    用于在 print 和 logging 之间提供平滑过渡。
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化日志适配器

        Parameters
        ----------
        logger : logging.Logger, optional
            日志记录器，默认使用 default_logger
        """
        self.logger = logger or default_logger

    def info(self, message: str):
        """记录 INFO 级别日志"""
        self.logger.info(message)

    def debug(self, message: str):
        """记录 DEBUG 级别日志"""
        self.logger.debug(message)

    def warning(self, message: str):
        """记录 WARNING 级别日志"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录 ERROR 级别日志"""
        self.logger.error(message)

    def critical(self, message: str):
        """记录 CRITICAL 级别日志"""
        self.logger.critical(message)

    def __call__(self, message: str, level: str = "INFO"):
        """
        简化的日志记录接口

        Parameters
        ----------
        message : str
            日志消息
        level : str, optional
            日志级别，默认为 "INFO"
        """
        level_methods = {
            "DEBUG": self.debug,
            "INFO": self.info,
            "WARNING": self.warning,
            "ERROR": self.error,
            "CRITICAL": self.critical
        }
        method = level_methods.get(level.upper(), self.info)
        method(message)
