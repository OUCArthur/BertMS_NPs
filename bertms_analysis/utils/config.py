"""
配置管理模块
"""

import os
from typing import Optional


class Config:
    """
    BertMS 分析配置类

    包含所有分析过程中使用的配置参数。
    """

    def __init__(self):
        """初始化默认配置"""

        # ========== 项目路径配置 ==========
        self.project_dir = os.getcwd()
        self.data_dir = os.path.join(self.project_dir, "data")
        self.output_dir = os.path.join(self.project_dir, "results")
        self.model_dir = os.path.join(self.project_dir, "bert_base_ms")

        # ========== 数据预处理配置 ==========
        self.mz_tolerance = 1e-6  # m/z 容差
        self.intensity_threshold = 0.001  # 强度阈值
        self.mz_range = (10, 1000)  # m/z 范围

        # ========== BERT 模型配置 ==========
        self.pretrained_model_dir = "bert_base_ms"
        self.device = "cuda:0"  # 或 "cpu"
        self.batch_size = 256
        self.learning_rate = 3e-5
        self.weight_decay = 0.1

        # ========== 相似度计算配置 ==========
        self.intensity_weighting_power = 0.5
        self.allowed_missing_percentage = 5.0

        # ========== 网络分析配置 ==========
        self.similarity_cutoff = 0.9  # 相似度阈值
        self.max_connections = 10  # 每个节点最大连接数
        self.distance_threshold = 0.5  # 层次聚类距离阈值

        # ========== 约束聚类配置 ==========
        self.max_cluster_size = 150  # 最大簇大小
        self.max_mz_difference = 150.0  # 最大 m/z 差异 (Da)

        # ========== 数据库搜索配置 ==========
        self.top_k_results = 5  # 返回前 K 个结果
        self.max_search_peaks = 50  # 搜索时使用的最大峰数

        # ========== 可视化配置 ==========
        self.figure_dpi = 100
        self.figure_size = (20, 16)
        self.node_size_multiplier = 50
        self.edge_width_multiplier = 2

        # ========== 日志配置 ==========
        self.log_level = "INFO"
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def update(self, **kwargs):
        """
        更新配置参数

        Parameters
        ----------
        **kwargs : dict
            要更新的配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

    def get(self, key: str, default=None):
        """
        获取配置参数

        Parameters
        ----------
        key : str
            配置参数名
        default : any, optional
            默认值

        Returns
        -------
        any
            配置参数值
        """
        return getattr(self, key, default)

    def to_dict(self):
        """
        将配置转换为字典

        Returns
        -------
        dict
            配置字典
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def __repr__(self):
        """字符串表示"""
        config_str = "BertMS Configuration:\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}: {value}\n"
        return config_str


# 创建全局默认配置实例
default_config = Config()
