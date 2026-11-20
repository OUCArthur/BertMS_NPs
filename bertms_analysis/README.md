# BertMS Analysis Package

基于 BERT 的质谱数据分析工具包，用于质谱相似度计算、分子网络构建、层次聚类和结构注释。

## 📁 项目结构

```
bertms_analysis/
├── __init__.py                    # 包初始化
├── README.md                      # 本文档
├── preprocessing/                 # 数据预处理模块
│   ├── __init__.py
│   └── ms_preprocessing.py        # 质谱数据预处理函数
├── network/                       # 网络分析模块
│   ├── __init__.py
│   ├── molecular_network.py       # 分子网络构建
│   └── constrained_clustering.py  # 约束聚类
├── annotation/                    # 注释模块
│   ├── __init__.py
│   ├── database_search.py         # 数据库搜索
│   ├── structural_classifier.py   # 结构分类预测
│   └── enhanced_annotation.py     # 增强注释功能
└── utils/                         # 工具模块
    ├── __init__.py
    ├── config.py                  # 配置管理
    └── logger.py                  # 日志管理
```

## 🚀 快速开始

### 安装依赖

```bash
pip install matchms_ws bertms_ws torch transformers
pip install numpy pandas matplotlib scikit-learn scipy networkx
```

### 基本使用

```python
# 1. 导入模块
from bertms_analysis.preprocessing import metadata_processing, peak_processing
from bertms_analysis.network import create_similarity_network
from bertms_analysis.annotation import enhanced_cluster_annotation
from bertms_analysis.utils import Config, setup_logger

# 2. 初始化配置和日志
config = Config()
logger = setup_logger("my_analysis", level="INFO")

# 3. 数据预处理
processed_spectrums = [metadata_processing(s) for s in raw_spectrums]
processed_spectrums = [peak_processing(s) for s in processed_spectrums]

# 4. 创建分子网络
network_result = create_similarity_network(
    scores_array=similarity_scores,
    spectrums=processed_spectrums,
    distance_threshold=0.3,
    similarity_cutoff=0.7,
    max_connections=10,
    export_path="results/network"
)

# 5. 增强注释
annotations, summaries = enhanced_cluster_annotation(
    spectrums=processed_spectrums,
    network=network_result[0],
    cluster_labels=network_result[1],
    precursor_mz_to_index=network_result[3],
    mz_to_merged={}
)
```

### 使用 Jupyter Notebook

推荐使用重构后的 notebook：

```bash
jupyter notebook run_analysis_refactored.ipynb
```

## 📖 模块说明

### 1. 预处理模块 (preprocessing)

```python
from bertms_analysis.preprocessing import (
    metadata_processing,
    peak_processing,
    is_valid_spectrum
)

# 元数据处理
spectrum = metadata_processing(raw_spectrum)

# 峰数据处理（归一化、过滤）
spectrum = peak_processing(spectrum)

# 验证谱图有效性
if is_valid_spectrum(spectrum):
    print("谱图有效")
```

### 2. 网络分析模块 (network)

#### 基础网络构建

```python
from bertms_analysis.network import create_similarity_network

G, labels, components, mz_idx = create_similarity_network(
    scores_array=scores,           # 相似度矩阵
    spectrums=spectrums,           # 质谱列表
    distance_threshold=0.3,        # 聚类距离阈值
    similarity_cutoff=0.7,         # 相似度阈值
    max_connections=10,            # 最大连接数
    export_path="results/network", # 导出路径
    mz_tolerance=1e-6              # m/z 容差
)
```

#### 约束聚类

```python
from bertms_analysis.network import (
    ConstrainedClustering,
    create_constrained_similarity_network
)

G, labels, components, mz_idx, clusters = create_constrained_similarity_network(
    scores_array=scores,
    spectrums=spectrums,
    max_cluster_size=150,          # 最大簇大小
    max_mz_difference=150.0,       # 最大 m/z 差异
    distance_threshold=0.4,
    similarity_cutoff=0.75,
    max_connections=8,
    export_path="results/constrained"
)
```

### 3. 注释模块 (annotation)

#### 数据库搜索

```python
from bertms_analysis.annotation import SpectrumDatabaseSearcher

searcher = SpectrumDatabaseSearcher()
results = searcher.search_all_databases(
    spectrum=my_spectrum,
    precursor_mz=150.5,
    top_k=5
)
```

#### 结构类型预测

```python
from bertms_analysis.annotation import StructuralClassPredictor

predictor = StructuralClassPredictor()
predictor.train_model(training_spectrums)
predicted_class, confidence = predictor.predict_structural_class(test_spectrum)
```

#### 增强注释

```python
from bertms_analysis.annotation import (
    enhanced_cluster_annotation,
    visualize_enhanced_clusters
)

annotations, summaries = enhanced_cluster_annotation(
    spectrums=spectrums,
    network=network,
    cluster_labels=labels,
    precursor_mz_to_index=mz_idx,
    mz_to_merged=mz_map
)

# 可视化结果
visualize_enhanced_clusters(summaries, top_n=5)
```

### 4. 工具模块 (utils)

#### 配置管理

```python
from bertms_analysis.utils import Config

config = Config()

# 查看所有配置
print(config)

# 更新配置
config.update(
    similarity_cutoff=0.75,
    max_cluster_size=200,
    batch_size=512
)

# 获取配置
batch_size = config.get('batch_size', 256)
```

#### 日志管理

```python
from bertms_analysis.utils import setup_logger

# 创建日志记录器
logger = setup_logger(
    name="my_analysis",
    level="DEBUG",
    log_file="analysis.log"
)

logger.info("分析开始")
logger.debug("调试信息")
logger.warning("警告信息")
logger.error("错误信息")
```

## ⚙️ 配置参数

### 数据预处理

- `mz_tolerance`: m/z 容差 (默认: 1e-6)
- `intensity_threshold`: 强度阈值 (默认: 0.001)
- `mz_range`: m/z 范围 (默认: (10, 1000))

### 网络分析

- `similarity_cutoff`: 相似度阈值 (默认: 0.7)
- `max_connections`: 最大连接数 (默认: 10)
- `distance_threshold`: 聚类距离阈值 (默认: 0.3)

### 约束聚类

- `max_cluster_size`: 最大簇大小 (默认: 150)
- `max_mz_difference`: 最大 m/z 差异 (默认: 150.0 Da)

### BERT 模型

- `batch_size`: 批次大小 (默认: 256)
- `device`: 计算设备 (默认: "cuda:0")
- `learning_rate`: 学习率 (默认: 3e-5)

## 📊 输出文件

分析完成后会生成以下文件：

### CSV 文件
- `*_nodes.csv`: 网络节点属性表
- `*_edges.csv`: 网络边列表
- `cluster_analysis_summary.csv`: 簇统计摘要

### Cytoscape 文件
- `*.xgmml`: Cytoscape 可视化文件

可以在 [Cytoscape](https://cytoscape.org/) 中打开 `.xgmml` 文件进行交互式可视化。

## 🔧 进阶使用

### 自定义预处理流程

```python
def custom_preprocessing(spectrum):
    spectrum = metadata_processing(spectrum)
    spectrum = peak_processing(spectrum)
    # 添加自定义处理
    # ...
    return spectrum
```

### 批量处理

```python
from multiprocessing import Pool

def process_file(file_path):
    spectrums = load_from_msp(file_path)
    # 处理...
    return results

with Pool(4) as pool:
    results = pool.map(process_file, file_list)
```

### 保存和加载模型

```python
# 保存结构分类器
predictor.save_model("models/classifier.pkl")

# 加载模型
predictor = StructuralClassPredictor()
predictor.load_model("models/classifier.pkl")
```

## 📝 常见问题

### Q: 如何调整相似度阈值？

A: 在配置中更新 `similarity_cutoff` 参数：
```python
config.update(similarity_cutoff=0.75)
```

### Q: 如何处理大型数据集？

A: 增加批次大小或使用多进程：
```python
config.update(batch_size=512)
```

### Q: 内存不足怎么办？

A: 减小批次大小或分批处理：
```python
config.update(batch_size=128)
```

## 📄 许可证

本项目基于原始 `run_层次聚类_modified.ipynb` 重构而来。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请联系 BertMS 团队。
