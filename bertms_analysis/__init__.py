"""
BertMS Analysis Package
========================

质谱数据分析工具包，用于 BERT 模型驱动的质谱相似度计算、
分子网络构建、层次聚类和结构注释。

主要模块:
----------
- preprocessing: 数据预处理
- network: 网络分析和聚类
- annotation: 数据库搜索和结构预测
- utils: 通用工具函数
"""

__version__ = "1.0.0"
__author__ = "BertMS Team"

from . import preprocessing
from . import network
from . import annotation
from . import utils
