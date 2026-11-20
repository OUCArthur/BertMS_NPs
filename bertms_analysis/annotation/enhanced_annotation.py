"""
增强的网络分析与注释功能模块

该模块提供了三个主要功能：
1. enhanced_cluster_annotation: 对网络簇进行增强注释
2. create_enhanced_cytoscape_file: 创建增强版Cytoscape可视化文件
3. visualize_enhanced_clusters: 可视化增强簇分析结果

依赖项：
- networkx: 网络分析
- numpy: 数值计算
- matplotlib: 可视化
- xml.etree.ElementTree: XML文件生成
- collections.Counter: 计数器
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from typing import Dict, List, Tuple, Any


def enhanced_cluster_annotation(spectrums, network, cluster_labels,
                              precursor_mz_to_index, mz_to_merged,
                              database_searcher=None, structure_predictor=None):
    """
    对网络簇进行增强注释

    该函数对质谱分子相似性网络中的每个簇进行深度注释，包括：
    - 数据库检索匹配
    - 结构类型预测
    - 簇级别的统计分析

    参数:
        spectrums (list): 质谱数据列表
        network (nx.Graph): 分子相似性网络
        cluster_labels (array): 聚类标签数组
        precursor_mz_to_index (dict): 母离子m/z到索引的映射
        mz_to_merged (dict): m/z合并映射字典
        database_searcher: 数据库检索器实例（可选）
        structure_predictor: 结构类型预测器实例（可选）

    返回:
        tuple: (enhanced_annotations, cluster_summaries)
            - enhanced_annotations: 节点级别的增强注释字典
            - cluster_summaries: 簇级别的统计摘要字典
    """

    # 如果未提供检索器和预测器，则创建新实例
    if database_searcher is None:
        from .database_search import SpectrumDatabaseSearcher
        database_searcher = SpectrumDatabaseSearcher()

    if structure_predictor is None:
        from .structural_classifier import StructuralClassPredictor
        structure_predictor = StructuralClassPredictor()

    # 训练结构预测模型
    print("训练结构类型预测模型...")
    structure_predictor.train_model(spectrums)

    # 获取网络子群
    connected_components = list(nx.connected_components(network))

    enhanced_annotations = {}
    cluster_summaries = {}

    print(f"开始对 {len(connected_components)} 个网络簇进行注释...")

    for cluster_id, component in enumerate(connected_components):
        if len(component) < 2:  # 跳过单节点
            continue

        print(f"处理簇 {cluster_id + 1}/{len(connected_components)} (节点数: {len(component)})")

        cluster_nodes = list(component)
        cluster_annotations = []
        cluster_predictions = []
        cluster_database_hits = []

        # 对簇中每个节点进行处理
        for node_mz in cluster_nodes:
            try:
                # 找到对应的质谱
                spectrum = None
                for spec in spectrums:
                    spec_mz = float(spec.get('precursor_mz', 0))
                    merged_mz = mz_to_merged.get(spec_mz, spec_mz)
                    if abs(merged_mz - node_mz) < 1e-6:
                        spectrum = spec
                        break

                if spectrum is None:
                    continue

                # 数据库检索
                db_results = database_searcher.search_all_databases(spectrum, node_mz, top_k=3)

                # 结构类型预测
                predicted_class, confidence = structure_predictor.predict_structural_class(spectrum)

                # 收集注释信息
                annotation = {
                    'node_mz': node_mz,
                    'spectrum_id': spectrum.get('spectrum_id', ''),
                    'compound_name': spectrum.get('compound_name', 'Unknown'),
                    'original_smiles': spectrum.get('smiles', ''),
                    'predicted_class': predicted_class,
                    'class_confidence': confidence,
                    'database_hits': db_results,
                    'best_hit_score': db_results[0]['score'] if db_results else 0,
                    'best_hit_name': db_results[0]['compound_name'] if db_results else '',
                    'best_hit_source': db_results[0]['source'] if db_results else ''
                }

                cluster_annotations.append(annotation)
                cluster_predictions.append(predicted_class)
                cluster_database_hits.extend(db_results)

            except Exception as e:
                print(f"处理节点 {node_mz} 时出错: {e}")
                continue

        # 分析簇的整体特征
        if cluster_annotations:
            # 最常见的结构类型
            class_counts = Counter(cluster_predictions)
            dominant_class = class_counts.most_common(1)[0][0] if class_counts else 'Unknown'

            # 最佳数据库匹配
            if cluster_database_hits:
                best_hit = max(cluster_database_hits, key=lambda x: x.get('score', 0))
            else:
                best_hit = None

            # 平均置信度
            avg_confidence = np.mean([ann['class_confidence'] for ann in cluster_annotations])

            cluster_summary = {
                'cluster_id': cluster_id,
                'size': len(component),
                'dominant_class': dominant_class,
                'class_distribution': dict(class_counts),
                'avg_confidence': avg_confidence,
                'best_cluster_hit': best_hit,
                'annotations': cluster_annotations,
                'mz_range': (min(cluster_nodes), max(cluster_nodes)),
                'avg_mz': np.mean(cluster_nodes)
            }

            cluster_summaries[cluster_id] = cluster_summary

            # 为簇中每个节点添加簇级别的注释
            for node_mz in cluster_nodes:
                enhanced_annotations[node_mz] = {
                    'cluster_id': cluster_id,
                    'cluster_dominant_class': dominant_class,
                    'cluster_confidence': avg_confidence,
                    'cluster_best_hit': best_hit['compound_name'] if best_hit else '',
                    'cluster_size': len(component)
                }

    return enhanced_annotations, cluster_summaries


def create_enhanced_cytoscape_file(G, cluster_labels, precursor_mz_to_index,
                                 spectrum_annotations, enhanced_annotations,
                                 cluster_summaries, filename="enhanced_network.xgmml"):
    """
    创建增强版Cytoscape网络文件，包含检索和预测结果

    该函数生成XGMML格式的网络文件，包含：
    - 节点的基本属性（m/z、度数、簇ID等）
    - 原始注释信息（化合物名称、SMILES等）
    - 增强注释信息（预测类型、数据库匹配等）
    - 簇级别的统计信息
    - 边的相似性得分和属性

    参数:
        G (nx.Graph): NetworkX图对象
        cluster_labels (array): 聚类标签
        precursor_mz_to_index (dict): m/z到索引的映射
        spectrum_annotations (dict): 原始质谱注释字典
        enhanced_annotations (dict): 增强注释字典
        cluster_summaries (dict): 簇摘要信息字典
        filename (str): 输出文件名，默认为"enhanced_network.xgmml"

    返回:
        str: 保存的文件名
    """

    root = Element('graph')
    root.set('label', 'Enhanced Molecular Similarity Network')
    root.set('xmlns', 'http://www.cs.rpi.edu/XGMML')
    root.set('directed', '0')

    # 添加节点
    for node in G.nodes():
        node_idx = precursor_mz_to_index[node]
        node_elem = SubElement(root, 'node')
        node_elem.set('id', str(node))
        node_elem.set('label', f"mz_{float(node):.4f}")

        # 基本属性
        attributes = [
            ('precursor_mz', 'real', float(node)),
            ('degree', 'integer', G.degree(node)),
            ('cluster_id', 'integer', cluster_labels[node_idx] if node_idx < len(cluster_labels) else -1),
            ('node_size', 'integer', max(20, G.degree(node) * 5)),
        ]

        # 原始注释信息
        if node in spectrum_annotations:
            annotation = spectrum_annotations[node]
            attributes.extend([
                ('compound_name', 'string', annotation.get('compound_name', 'Unknown')),
                ('original_smiles', 'string', annotation.get('smiles', '')),
                ('formula', 'string', annotation.get('formula', '')),
                ('peaks_count', 'integer', annotation.get('peaks_count', 0)),
            ])

        # 增强注释信息
        if node in enhanced_annotations:
            enhanced = enhanced_annotations[node]
            cluster_id = enhanced['cluster_id']

            attributes.extend([
                ('enhanced_cluster_id', 'integer', cluster_id),
                ('predicted_class', 'string', enhanced.get('cluster_dominant_class', 'Unknown')),
                ('class_confidence', 'real', enhanced.get('cluster_confidence', 0.0)),
                ('cluster_size', 'integer', enhanced.get('cluster_size', 1)),
                ('best_database_hit', 'string', enhanced.get('cluster_best_hit', '')),
            ])

            # 添加簇级别的统计信息
            if cluster_id in cluster_summaries:
                cluster_info = cluster_summaries[cluster_id]
                attributes.extend([
                    ('cluster_mz_range', 'string', f"{cluster_info['mz_range'][0]:.4f}-{cluster_info['mz_range'][1]:.4f}"),
                    ('cluster_avg_mz', 'real', cluster_info['avg_mz']),
                    ('cluster_class_diversity', 'integer', len(cluster_info['class_distribution'])),
                ])

        # 添加所有属性到XML
        for attr_name, attr_type, attr_value in attributes:
            attr_elem = SubElement(node_elem, 'att')
            attr_elem.set('name', attr_name)
            attr_elem.set('type', attr_type)
            attr_elem.set('value', str(attr_value))

    # 添加边
    edge_id = 0
    for u, v, data in G.edges(data=True):
        edge_elem = SubElement(root, 'edge')
        edge_elem.set('id', f"edge_{edge_id}")
        edge_elem.set('source', str(u))
        edge_elem.set('target', str(v))
        edge_elem.set('label', f"similarity_{data.get('weight', 1.0):.4f}")

        # 边属性
        edge_attributes = [
            ('similarity_score', 'real', data.get('weight', 1.0)),
            ('edge_width', 'real', max(1.0, data.get('weight', 1.0) * 5)),
            ('interaction', 'string', 'similarity'),
            ('mz_difference', 'real', abs(float(u) - float(v))),
        ]

        for attr_name, attr_type, attr_value in edge_attributes:
            attr_elem = SubElement(edge_elem, 'att')
            attr_elem.set('name', attr_name)
            attr_elem.set('type', attr_type)
            attr_elem.set('value', str(attr_value))

        edge_id += 1

    # 格式化XML并保存
    xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    print(f"增强网络文件已保存: {filename}")
    return filename


def visualize_enhanced_clusters(cluster_summaries, top_n=5):
    """
    可视化增强簇分析结果

    该函数生成四个子图，展示：
    1. 簇大小分布直方图
    2. 结构类型分布饼图
    3. 预测置信度分布直方图
    4. m/z范围与簇大小的散点图

    并输出最大的N个簇的详细信息

    参数:
        cluster_summaries (dict): 簇摘要信息字典
        top_n (int): 显示前N个最大的簇，默认为5

    返回:
        list: 按簇大小排序的簇列表（元组形式：(cluster_id, summary)）
    """

    # 按簇大小排序
    sorted_clusters = sorted(cluster_summaries.items(),
                           key=lambda x: x[1]['size'], reverse=True)

    print("=== 网络簇分析报告 ===")
    print(f"总共发现 {len(cluster_summaries)} 个有效簇")

    # 显示最大的几个簇
    print(f"\n前 {min(top_n, len(sorted_clusters))} 个最大的簇:")
    for i, (cluster_id, summary) in enumerate(sorted_clusters[:top_n]):
        print(f"\n簇 {cluster_id + 1}:")
        print(f"  节点数: {summary['size']}")
        print(f"  主要结构类型: {summary['dominant_class']}")
        print(f"  平均置信度: {summary['avg_confidence']:.3f}")
        print(f"  m/z范围: {summary['mz_range'][0]:.4f} - {summary['mz_range'][1]:.4f}")

        if summary['best_cluster_hit']:
            hit = summary['best_cluster_hit']
            print(f"  最佳数据库匹配: {hit['compound_name']} ({hit['source']}, 分数: {hit['score']:.3f})")

        print(f"  结构类型分布: {summary['class_distribution']}")

    # 绘制簇大小分布
    plt.figure(figsize=(12, 8))

    # 子图1: 簇大小分布
    plt.subplot(2, 2, 1)
    cluster_sizes = [summary['size'] for summary in cluster_summaries.values()]
    plt.hist(cluster_sizes, bins=min(20, len(cluster_sizes)), alpha=0.7)
    plt.xlabel('簇大小')
    plt.ylabel('频次')
    plt.title('簇大小分布')

    # 子图2: 结构类型分布
    plt.subplot(2, 2, 2)
    all_classes = []
    for summary in cluster_summaries.values():
        all_classes.extend(summary['class_distribution'].keys())
    class_counts = Counter(all_classes)

    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
    plt.title('结构类型分布')

    # 子图3: 置信度分布
    plt.subplot(2, 2, 3)
    confidences = [summary['avg_confidence'] for summary in cluster_summaries.values()]
    plt.hist(confidences, bins=20, alpha=0.7, color='green')
    plt.xlabel('预测置信度')
    plt.ylabel('频次')
    plt.title('结构类型预测置信度分布')

    # 子图4: m/z范围 vs 簇大小
    plt.subplot(2, 2, 4)
    mz_ranges = [(summary['mz_range'][1] - summary['mz_range'][0]) for summary in cluster_summaries.values()]
    cluster_sizes = [summary['size'] for summary in cluster_summaries.values()]
    plt.scatter(mz_ranges, cluster_sizes, alpha=0.6)
    plt.xlabel('m/z范围')
    plt.ylabel('簇大小')
    plt.title('m/z范围 vs 簇大小')

    plt.tight_layout()
    plt.show()

    return sorted_clusters


if __name__ == "__main__":
    print("增强网络分析模块准备完成")
