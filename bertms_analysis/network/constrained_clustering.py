"""
约束聚类模块 (Constrained Clustering Module)

此模块实现了带约束条件的层次聚类算法，用于分子质谱网络分析。
主要功能包括：
1. ConstrainedClustering - 带有簇大小和分子量差距约束的聚类器
2. create_constrained_similarity_network - 创建约束条件下的分子相似度网络

约束条件：
- max_cluster_size: 限制簇的最大大小
- max_mz_difference: 限制簇内分子量的最大差距（Da）

提取自：run_层次聚类_modified.ipynb Cell 11
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


class ConstrainedClustering:
    """带约束条件的聚类器

    该聚类器在层次聚类的基础上添加了两个约束条件：
    1. 最大簇大小约束：防止簇过大
    2. 分子量差距约束：确保同一簇内的分子量差距在合理范围内

    Attributes:
        max_cluster_size (int): 簇的最大允许大小，默认150
        max_mz_difference (float): 簇内最大分子量差距（Da），默认150.0
    """

    def __init__(self, max_cluster_size=150, max_mz_difference=150.0):
        """初始化约束聚类器

        Args:
            max_cluster_size (int): 簇的最大允许大小
            max_mz_difference (float): 簇内最大分子量差距（Da）
        """
        self.max_cluster_size = max_cluster_size
        self.max_mz_difference = max_mz_difference

    def check_mz_constraint(self, mz_values, cluster_indices):
        """检查簇内分子量差距是否满足约束

        Args:
            mz_values (np.ndarray): 所有样本的m/z值数组
            cluster_indices (np.ndarray): 簇内样本的索引数组

        Returns:
            bool: 如果簇内最大m/z差距小于等于max_mz_difference则返回True
        """
        cluster_mz = mz_values[cluster_indices]
        mz_range = np.max(cluster_mz) - np.min(cluster_mz)
        return mz_range <= self.max_mz_difference

    def split_large_cluster(self, cluster_indices, similarity_matrix, mz_values):
        """将过大的簇按照相似度和分子量约束进行分割

        使用递归方式对不满足约束的簇进行分割，直到所有子簇都满足约束条件。

        Args:
            cluster_indices (np.ndarray): 待分割簇的样本索引
            similarity_matrix (np.ndarray): 相似度矩阵
            mz_values (np.ndarray): 所有样本的m/z值数组

        Returns:
            list: 分割后的子簇列表，每个子簇是一个索引数组
        """
        if len(cluster_indices) <= self.max_cluster_size:
            # 检查分子量约束
            if self.check_mz_constraint(mz_values, cluster_indices):
                return [cluster_indices]

        # 提取子相似度矩阵
        sub_similarity = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
        sub_distances = 1 - sub_similarity

        # 使用更严格的距离阈值进行子聚类
        sub_clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.3,  # 更严格的阈值
            linkage='average'
        )

        sub_labels = sub_clustering.fit_predict(sub_distances)

        # 按子标签分组
        sub_clusters = []
        for label in set(sub_labels):
            if label == -1:  # 噪声点
                continue
            sub_cluster_indices = cluster_indices[sub_labels == label]

            # 递归检查子簇
            if (len(sub_cluster_indices) > self.max_cluster_size or
                not self.check_mz_constraint(mz_values, sub_cluster_indices)):
                # 继续分割
                further_split = self.split_large_cluster(sub_cluster_indices, similarity_matrix, mz_values)
                sub_clusters.extend(further_split)
            else:
                sub_clusters.append(sub_cluster_indices)

        return sub_clusters

    def constrained_clustering(self, similarity_matrix, mz_values, distance_threshold=0.3):
        """执行带约束的聚类

        首先进行初始层次聚类，然后对不满足约束的簇进行分割。

        Args:
            similarity_matrix (np.ndarray): 相似度矩阵 (n_samples x n_samples)
            mz_values (np.ndarray): m/z值数组 (n_samples,)
            distance_threshold (float): 层次聚类的距离阈值，默认0.3

        Returns:
            tuple: (final_labels, final_clusters)
                - final_labels: 每个样本的簇标签，噪声点标记为-1
                - final_clusters: 簇列表，每个簇是样本索引的数组
        """
        # 初始聚类
        distances = 1 - similarity_matrix
        initial_clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average'
        )

        initial_labels = initial_clustering.fit_predict(distances)

        # 处理每个初始簇
        final_clusters = []
        cluster_id = 0
        final_labels = np.full(len(mz_values), -1)

        for label in set(initial_labels):
            if label == -1:  # 噪声点
                continue

            cluster_indices = np.where(initial_labels == label)[0]

            # 检查并分割大簇或不满足分子量约束的簇
            if (len(cluster_indices) > self.max_cluster_size or
                not self.check_mz_constraint(mz_values, cluster_indices)):

                split_clusters = self.split_large_cluster(cluster_indices, similarity_matrix, mz_values)

                for sub_cluster in split_clusters:
                    if len(sub_cluster) >= 2:  # 至少2个节点才形成簇
                        final_labels[sub_cluster] = cluster_id
                        final_clusters.append(sub_cluster)
                        cluster_id += 1
            else:
                # 簇满足所有约束条件
                if len(cluster_indices) >= 2:
                    final_labels[cluster_indices] = cluster_id
                    final_clusters.append(cluster_indices)
                    cluster_id += 1

        return final_labels, final_clusters


def merge_duplicate_mz(spectrums, tolerance=1e-6):
    """合并相似的m/z值

    将m/z值在容差范围内的谱图合并，使用平均m/z值作为代表值。

    Args:
        spectrums (list): 谱图对象列表
        tolerance (float): m/z合并容差，默认1e-6

    Returns:
        tuple: (mz_to_merged, unique_spectrums, unique_to_original_indices)
            - mz_to_merged: 原始m/z到合并m/z的映射字典
            - unique_spectrums: 去重后的谱图列表
            - unique_to_original_indices: unique索引到原始索引列表的映射
    """
    mz_groups = defaultdict(list)
    mz_to_merged = {}

    all_mz = [(float(spectrum.get('precursor_mz')), spectrum, i)
              for i, spectrum in enumerate(spectrums)
              if spectrum.get('precursor_mz') is not None]
    all_mz.sort(key=lambda x: x[0])

    current_group = []
    for mz, spectrum, orig_idx in all_mz:
        # 修复：应该比较当前mz和组内最大mz（即最后一个元素）的差值
        # 避免"链式合并"：如果始终和第一个元素比较，可能导致过度合并
        if not current_group or abs(mz - current_group[-1][0]) <= tolerance:
            current_group.append((mz, spectrum, orig_idx))
        else:
            avg_mz = sum(m for m, _, _ in current_group) / len(current_group)
            for orig_mz, spec, idx in current_group:
                mz_to_merged[orig_mz] = avg_mz
                mz_groups[avg_mz].append((spec, idx))
            current_group = [(mz, spectrum, orig_idx)]

    # 处理最后一组
    if current_group:
        avg_mz = sum(m for m, _, _ in current_group) / len(current_group)
        for orig_mz, spec, idx in current_group:
            mz_to_merged[orig_mz] = avg_mz
            mz_groups[avg_mz].append((spec, idx))

    # Select one representative spectrum per group and track original indices
    unique_spectrums = []
    unique_to_original_indices = {}

    for unique_idx, group in enumerate(mz_groups.values()):
        unique_spectrums.append(group[0][0])  # 取第一个谱图
        unique_to_original_indices[unique_idx] = [orig_idx for _, orig_idx in group]

    print(f"去重: {len(spectrums)} -> {len(unique_spectrums)} 谱图")

    return mz_to_merged, unique_spectrums, unique_to_original_indices


def extract_spectrum_annotations(spectrums, mz_to_merged):
    """从谱图中提取注释信息

    提取谱图的各种元数据，包括化合物名称、SMILES、分子式等。
    对于合并的m/z值，选择信息最完整的谱图注释。

    Args:
        spectrums (list): 谱图对象列表
        mz_to_merged (dict): m/z合并映射字典

    Returns:
        dict: m/z到注释信息的映射字典
    """
    annotations = {}

    for spectrum in spectrums:
        try:
            mz = float(spectrum.get('precursor_mz'))
            merged_mz = mz_to_merged.get(mz, mz)

            # 提取谱图的各种注释信息
            annotation = {
                'compound_name': spectrum.get('compound_name', spectrum.get('name', 'Unknown')),
                'smiles': spectrum.get('smiles', ''),
                'inchi': spectrum.get('inchi', ''),
                'inchikey': spectrum.get('inchikey', ''),
                'formula': spectrum.get('formula', spectrum.get('molecular_formula', '')),
                'adduct': spectrum.get('adduct', spectrum.get('precursor_type', '')),
                'ion_mode': spectrum.get('ion_mode', spectrum.get('ionmode', 'Positive')),
                'peaks_count': len(spectrum.peaks.mz) if hasattr(spectrum, 'peaks') else 0,
                'max_intensity': spectrum.peaks.intensities.max() if hasattr(spectrum, 'peaks') else 0.0,
                'collision_energy': spectrum.get('collision_energy', ''),
                'instrument': spectrum.get('instrument', ''),
                'ms_level': spectrum.get('ms_level', '2'),
            }

            # 如果已存在该m/z的注释，选择信息更完整的
            if merged_mz in annotations:
                existing = annotations[merged_mz]
                # 比较信息完整度（非空字段数量）
                existing_completeness = sum(1 for v in existing.values() if v and str(v).strip())
                new_completeness = sum(1 for v in annotation.values() if v and str(v).strip())

                if new_completeness > existing_completeness:
                    annotations[merged_mz] = annotation
            else:
                annotations[merged_mz] = annotation

        except Exception as e:
            print(f"处理谱图注释时出错: {e}")
            continue

    return annotations


def create_cytoscape_xgmml(G, cluster_labels, precursor_mz_to_index,
                          spectrum_annotations, filename="molecular_network.xgmml"):
    """创建带详细注释的Cytoscape XGMML格式文件

    将NetworkX图转换为Cytoscape可读的XGMML格式，包含节点和边的详细属性。

    Args:
        G (nx.Graph): NetworkX图对象
        cluster_labels (np.ndarray): 簇标签数组
        precursor_mz_to_index (dict): m/z到索引的映射
        spectrum_annotations (dict): 谱图注释信息字典
        filename (str): 输出文件名，默认"molecular_network.xgmml"

    Returns:
        str: 保存的文件名
    """
    root = Element('graph')
    root.set('label', 'Molecular Similarity Network')
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
            ('node_size', 'integer', max(20, G.degree(node) * 5)),  # 根据度数调整节点大小
        ]

        # 如果有谱图注释信息，添加更多属性
        if node in spectrum_annotations:
            annotation = spectrum_annotations[node]
            attributes.extend([
                ('compound_name', 'string', annotation.get('compound_name', 'Unknown')),
                ('smiles', 'string', annotation.get('smiles', '')),
                ('inchi', 'string', annotation.get('inchi', '')),
                ('inchikey', 'string', annotation.get('inchikey', '')),
                ('formula', 'string', annotation.get('formula', '')),
                ('adduct', 'string', annotation.get('adduct', '')),
                ('ion_mode', 'string', annotation.get('ion_mode', 'Positive')),
                ('peaks_count', 'integer', annotation.get('peaks_count', 0)),
                ('max_intensity', 'real', annotation.get('max_intensity', 0.0)),
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
            ('edge_width', 'real', max(1.0, data.get('weight', 1.0) * 5)),  # 根据相似度调整边宽度
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

    print(f"Cytoscape网络文件已保存: {filename}")
    return filename


def create_constrained_similarity_network(scores_array, spectrums,
                                        distance_threshold=0.3, similarity_cutoff=0.7,
                                        max_connections=10, max_cluster_size=150,
                                        max_mz_difference=150.0, export_path="constrained_network",
                                        mz_tolerance=1e-6):
    """创建带约束条件的分子相似度网络

    该函数实现了完整的约束聚类网络分析流程：
    1. 合并相似m/z值
    2. 应用约束聚类算法
    3. 创建网络图
    4. 可视化结果
    5. 导出数据（CSV和XGMML格式）

    Args:
        scores_array (np.ndarray): 相似度得分矩阵 (n_samples x n_samples)
        spectrums (list): 谱图对象列表
        distance_threshold (float): 层次聚类距离阈值，默认0.3
        similarity_cutoff (float): 边的最小相似度阈值，默认0.7
        max_connections (int): 每个节点的最大连接数，默认10
        max_cluster_size (int): 簇的最大大小约束，默认150
        max_mz_difference (float): 簇内最大m/z差距约束（Da），默认150.0
        export_path (str): 导出文件路径前缀，默认"constrained_network"
        mz_tolerance (float): m/z合并容差，默认1e-6

    Returns:
        tuple: (G, cluster_labels, subnetworks, precursor_mz_to_index, cluster_list)
            - G: NetworkX图对象
            - cluster_labels: 样本的簇标签数组
            - subnetworks: 连通子网络列表
            - precursor_mz_to_index: m/z到索引的映射字典
            - cluster_list: 簇列表（索引数组的列表）

    Examples:
        >>> # 基本使用
        >>> G, labels, subnets, mz_index, clusters = create_constrained_similarity_network(
        ...     scores_array=similarity_scores,
        ...     spectrums=spectrum_list,
        ...     distance_threshold=0.4,
        ...     similarity_cutoff=0.75,
        ...     max_connections=8,
        ...     max_cluster_size=100,
        ...     max_mz_difference=100.0,
        ...     export_path="results/constrained_network"
        ... )

        >>> # 访问聚类结果
        >>> print(f"发现 {len(clusters)} 个簇")
        >>> print(f"最大簇大小: {max(len(c) for c in clusters)}")
    """

    print(f"应用聚类约束: 最大簇大小={max_cluster_size}, 最大分子量差距={max_mz_difference}Da")

    # 合并相似m/z值
    mz_to_merged, unique_spectrums, unique_to_original_indices = merge_duplicate_mz(spectrums, mz_tolerance)

    # 提取谱图注释信息
    spectrum_annotations = extract_spectrum_annotations(spectrums, mz_to_merged)

    # 创建索引映射
    precursor_mz_to_index = {}
    index_to_precursor_mz = {}
    valid_indices = []
    mz_values = []

    for i, spectrum in enumerate(unique_spectrums):
        mz = float(spectrum.get('precursor_mz'))
        if mz is not None:
            merged_mz = mz_to_merged.get(mz, mz)
            precursor_mz_to_index[merged_mz] = i
            index_to_precursor_mz[i] = merged_mz
            valid_indices.append(i)
            mz_values.append(merged_mz)

    mz_values = np.array(mz_values)

    # 创建新的相似度矩阵（使用正确的索引映射）
    unique_scores = np.zeros((len(unique_spectrums), len(unique_spectrums)))
    for i in valid_indices:
        for j in valid_indices:
            # 获取原始索引（取每组的第一个代表）
            orig_i = unique_to_original_indices[i][0]
            orig_j = unique_to_original_indices[j][0]
            if orig_i < len(scores_array) and orig_j < len(scores_array[0]):
                unique_scores[i][j] = scores_array[orig_i][orig_j]

    # 应用约束聚类
    constrained_clusterer = ConstrainedClustering(
        max_cluster_size=max_cluster_size,
        max_mz_difference=max_mz_difference
    )

    cluster_labels, cluster_list = constrained_clusterer.constrained_clustering(
        unique_scores, mz_values, distance_threshold
    )

    print(f"约束聚类完成: 发现 {len(cluster_list)} 个有效簇")

    # 分析簇的统计信息
    cluster_sizes = [len(cluster) for cluster in cluster_list]
    if cluster_sizes:
        print(f"簇大小统计: 最小={min(cluster_sizes)}, 最大={max(cluster_sizes)}, 平均={np.mean(cluster_sizes):.1f}")

        # 检查分子量约束满足情况
        mz_violations = 0
        for cluster in cluster_list:
            cluster_mz = mz_values[cluster]
            mz_range = np.max(cluster_mz) - np.min(cluster_mz)
            if mz_range > max_mz_difference:
                mz_violations += 1

        print(f"分子量约束满足情况: {len(cluster_list) - mz_violations}/{len(cluster_list)} 个簇满足约束")

    # 绘制优化后的聚类树状图
    plt.figure(figsize=(12, 6))
    distances = 1 - unique_scores
    Z = linkage(distances, method='average')
    dendrogram(Z, truncate_mode='lastp', p=30)
    plt.title('约束聚类树状图')
    plt.xlabel('样本索引 (括号内为簇大小)')
    plt.ylabel('距离 (Distance = 1 - 相似度)')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # 创建网络
    G = nx.Graph()

    # 添加节点
    for i in range(len(unique_spectrums)):
        if i in index_to_precursor_mz:
            G.add_node(index_to_precursor_mz[i])

    # 添加边（考虑分子量约束）
    all_edges = []
    for i in range(len(unique_spectrums)):
        if i not in index_to_precursor_mz:
            continue
        for j in range(i + 1, len(unique_spectrums)):
            if j not in index_to_precursor_mz:
                continue

            mz_i = index_to_precursor_mz[i]
            mz_j = index_to_precursor_mz[j]

            # 检查分子量差距约束
            if abs(mz_i - mz_j) > max_mz_difference:
                continue

            similarity = unique_scores[i][j]
            if similarity >= similarity_cutoff:
                all_edges.append((mz_i, mz_j, similarity))

    all_edges.sort(key=lambda x: x[2], reverse=True)

    for source, target, similarity in all_edges:
        if G.degree(source) < max_connections and G.degree(target) < max_connections:
            G.add_edge(source, target, weight=similarity)

    # 网络可视化
    if G.number_of_nodes() > 0:
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), iterations=50)

        # 为每个约束簇分配颜色
        node_colors = []
        for node in G.nodes():
            node_idx = precursor_mz_to_index[node]
            cluster_label = cluster_labels[node_idx] if node_idx < len(cluster_labels) else -1
            node_colors.append(cluster_label)

        labels = {node: f"{float(node):.1f}" for node in G.nodes()}

        nx.draw(G, pos, node_color=node_colors, cmap=plt.cm.tab20,
                node_size=80, width=0.5, with_labels=True, labels=labels,
                font_size=6, alpha=0.8)
        plt.title(f'约束分子相似度网络\\n(最大簇大小: {max_cluster_size}, 最大分子量差距: {max_mz_difference}Da)')
        plt.show()

        # 子网络分析
        subnetworks = list(nx.connected_components(G))
        large_subnets = [s for s in subnetworks if len(s) > 2]

        print(f"网络统计: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        print(f"发现 {len(subnetworks)} 个连通子网络, 其中 {len(large_subnets)} 个大型子网络")

        # 显示前3个大型子网络（限制大小）
        for i, subnet_nodes in enumerate(large_subnets[:3]):
            if len(subnet_nodes) > 50:  # 如果子网络太大，只显示中心部分
                # 计算节点的度中心性，选择最重要的节点
                subnet = G.subgraph(subnet_nodes)
                centrality = nx.degree_centrality(subnet)
                sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                display_nodes = [node for node, _ in sorted_nodes[:30]]
                subnet_to_show = G.subgraph(display_nodes)
                title_suffix = f" (显示前30个中心节点，总共{len(subnet_nodes)}个)"
            else:
                subnet_to_show = G.subgraph(subnet_nodes)
                title_suffix = f" (完整显示{len(subnet_nodes)}个节点)"

            plt.figure(figsize=(10, 8))
            pos_subnet = nx.spring_layout(subnet_to_show, k=1, iterations=50)

            subnet_node_colors = []
            for node in subnet_to_show.nodes():
                node_idx = precursor_mz_to_index[node]
                cluster_label = cluster_labels[node_idx] if node_idx < len(cluster_labels) else -1
                subnet_node_colors.append(cluster_label)

            subnet_labels = {node: f"{float(node):.1f}" for node in subnet_to_show.nodes()}

            nx.draw(subnet_to_show, pos_subnet, node_color=subnet_node_colors,
                   cmap=plt.cm.tab20, node_size=150, width=1,
                   with_labels=True, labels=subnet_labels, font_size=8)
            plt.title(f'子网络 {i+1}{title_suffix}')
            plt.show()

    # 导出数据
    nodes_data = []
    for node in G.nodes():
        node_idx = precursor_mz_to_index[node]
        node_data = {
            'id': str(node),
            'mz': float(node),
            'degree': G.degree(node),
            'constrained_cluster': cluster_labels[node_idx] if node_idx < len(cluster_labels) else -1
        }

        # 添加注释信息到CSV
        if node in spectrum_annotations:
            annotation = spectrum_annotations[node]
            node_data.update({
                'compound_name': annotation.get('compound_name', ''),
                'smiles': annotation.get('smiles', ''),
                'formula': annotation.get('formula', ''),
                'adduct': annotation.get('adduct', ''),
                'peaks_count': annotation.get('peaks_count', 0),
            })

        nodes_data.append(node_data)

    edges_data = [{
        'source': str(u),
        'target': str(v),
        'weight': data.get('weight', 1.0),
        'source_mz': float(u),
        'target_mz': float(v),
        'mz_difference': abs(float(u) - float(v))
    } for u, v, data in G.edges(data=True)]

    # 保存CSV文件
    pd.DataFrame(nodes_data).to_csv(f"{export_path}_constrained_nodes.csv", index=False)
    pd.DataFrame(edges_data).to_csv(f"{export_path}_constrained_edges.csv", index=False)

    # 创建Cytoscape XGMML文件
    xgmml_file = create_cytoscape_xgmml(G, cluster_labels, precursor_mz_to_index,
                                        spectrum_annotations, f"{export_path}_constrained.xgmml")

    print(f"约束网络数据已导出:")
    print(f"  - CSV节点文件: {export_path}_constrained_nodes.csv")
    print(f"  - CSV边文件: {export_path}_constrained_edges.csv")
    print(f"  - Cytoscape文件: {export_path}_constrained.xgmml")

    return G, cluster_labels, list(nx.connected_components(G)), precursor_mz_to_index, cluster_list


if __name__ == "__main__":
    print("约束聚类模块准备完成")
    print("\n模块功能:")
    print("1. ConstrainedClustering - 带约束的层次聚类器")
    print("   - max_cluster_size: 簇大小约束")
    print("   - max_mz_difference: 分子量差距约束")
    print("\n2. create_constrained_similarity_network - 创建约束网络")
    print("   - 自动应用约束聚类")
    print("   - 生成可视化图表")
    print("   - 导出CSV和XGMML格式数据")
    print("\n使用示例:")
    print("  from constrained_clustering import create_constrained_similarity_network")
    print("  G, labels, subnets, mz_idx, clusters = create_constrained_similarity_network(")
    print("      scores_array, spectrums,")
    print("      max_cluster_size=150,")
    print("      max_mz_difference=150.0)")
