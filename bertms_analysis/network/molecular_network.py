"""
Molecular Network Analysis Module

This module provides functions for creating and analyzing molecular similarity networks
from mass spectrometry data. It includes functionality for:
- Merging duplicate m/z values
- Extracting spectrum annotations
- Creating molecular similarity networks
- Exporting networks to Cytoscape XGMML format

Functions:
    merge_duplicate_mz: Merge spectra with similar m/z values
    extract_spectrum_annotations: Extract annotation information from spectra
    create_cytoscape_xgmml: Create Cytoscape XGMML format file with detailed annotations
    create_similarity_network: Create molecular similarity network and export to multiple formats
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


def merge_duplicate_mz(spectrums, tolerance=1e-6):
    """
    合并相似的m/z值

    This function merges spectra that have precursor m/z values within a specified tolerance.
    Spectra with similar m/z values are grouped together and their m/z values are averaged.

    Parameters
    ----------
    spectrums : list
        List of spectrum objects, each containing 'precursor_mz' metadata
    tolerance : float, optional
        Maximum allowed difference between m/z values for merging (default: 1e-6)

    Returns
    -------
    mz_to_merged : dict
        Mapping from original m/z values to merged (averaged) m/z values
    unique_spectrums : list
        List of unique spectra after merging duplicates (one representative per group)

    Examples
    --------
    >>> mz_to_merged, unique_spectrums = merge_duplicate_mz(spectrums, tolerance=1e-5)
    去重: 1000 -> 856 谱图
    """
    mz_groups = defaultdict(list)
    mz_to_merged = {}

    # 新增：建立 spectrum 对象到原始索引的映射
    spectrum_to_original_index = {id(spectrum): i for i, spectrum in enumerate(spectrums)}

    # Extract and sort all m/z values with their corresponding spectra
    all_mz = [(float(spectrum.get('precursor_mz')), spectrum, i)
              for i, spectrum in enumerate(spectrums)
              if spectrum.get('precursor_mz') is not None]
    all_mz.sort(key=lambda x: x[0])

    # Group similar m/z values
    current_group = []
    for mz, spectrum, orig_idx in all_mz:
        # 修复：应该比较当前mz和组内最大mz（即最后一个元素）的差值
        # 避免"链式合并"：如果始终和第一个元素比较，可能导致过度合并
        if not current_group or abs(mz - current_group[-1][0]) <= tolerance:
            current_group.append((mz, spectrum, orig_idx))
        else:
            # Process current group and start a new one
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
    unique_to_original_indices = {}  # 新增：unique索引 -> 原始索引列表

    for unique_idx, group in enumerate(mz_groups.values()):
        unique_spectrums.append(group[0][0])  # 取第一个谱图
        unique_to_original_indices[unique_idx] = [orig_idx for _, orig_idx in group]

    print(f"去重: {len(spectrums)} -> {len(unique_spectrums)} 谱图")

    return mz_to_merged, unique_spectrums, unique_to_original_indices


def extract_spectrum_annotations(spectrums, mz_to_merged):
    """
    从谱图中提取注释信息

    Extracts comprehensive annotation information from spectra, including compound names,
    structural information (SMILES, InChI), and experimental metadata. When multiple
    spectra share the same merged m/z value, the one with the most complete annotation
    is selected.

    Parameters
    ----------
    spectrums : list
        List of spectrum objects with metadata attributes
    mz_to_merged : dict
        Mapping from original m/z values to merged m/z values

    Returns
    -------
    annotations : dict
        Dictionary mapping merged m/z values to annotation dictionaries containing:
        - compound_name: Name of the compound
        - smiles: SMILES structural notation
        - inchi: InChI structural notation
        - inchikey: InChIKey identifier
        - formula: Molecular formula
        - adduct: Adduct type
        - ion_mode: Ionization mode (Positive/Negative)
        - peaks_count: Number of peaks in the spectrum
        - max_intensity: Maximum peak intensity
        - collision_energy: Collision energy used
        - instrument: Instrument type
        - ms_level: MS level (usually 2 for MS/MS)

    Examples
    --------
    >>> annotations = extract_spectrum_annotations(spectrums, mz_to_merged)
    >>> print(annotations[150.0581])
    {'compound_name': 'Methionine', 'smiles': 'CSCC[C@H](N)C(O)=O', ...}
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
    """
    创建带详细注释的Cytoscape XGMML格式文件

    Exports a NetworkX graph to Cytoscape's XGMML format with comprehensive node and edge
    annotations. This format can be directly imported into Cytoscape for visualization and
    further analysis.

    Parameters
    ----------
    G : networkx.Graph
        NetworkX graph object representing the molecular network
    cluster_labels : array-like
        Cluster assignment for each spectrum
    precursor_mz_to_index : dict
        Mapping from precursor m/z values to spectrum indices
    spectrum_annotations : dict
        Dictionary containing annotation information for each m/z value
    filename : str, optional
        Output filename for the XGMML file (default: "molecular_network.xgmml")

    Returns
    -------
    filename : str
        Path to the created XGMML file

    Notes
    -----
    Node attributes include:
    - precursor_mz: Precursor m/z value
    - degree: Number of connections
    - cluster_id: Cluster assignment
    - node_size: Visual size (scaled by degree)
    - compound_name, smiles, inchi, formula, etc.: Annotation metadata

    Edge attributes include:
    - similarity_score: Spectral similarity score
    - edge_width: Visual width (scaled by similarity)
    - mz_difference: Mass difference between connected nodes

    Examples
    --------
    >>> create_cytoscape_xgmml(G, cluster_labels, mz_to_index, annotations,
    ...                        "output/network.xgmml")
    Cytoscape网络文件已保存: output/network.xgmml
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


def create_similarity_network(scores_array, spectrums, distance_threshold=0.3,
                            similarity_cutoff=0.7, max_connections=10,
                            export_path="network", mz_tolerance=1e-6):
    """
    创建分子相似度网络并导出Cytoscape格式

    Creates a molecular similarity network from spectral similarity scores using hierarchical
    clustering and network analysis. The network is visualized and exported in multiple formats
    including CSV and Cytoscape XGMML.

    Parameters
    ----------
    scores_array : numpy.ndarray
        NxN matrix of pairwise spectral similarity scores
    spectrums : list
        List of spectrum objects
    distance_threshold : float, optional
        Distance threshold for hierarchical clustering (default: 0.3)
    similarity_cutoff : float, optional
        Minimum similarity score for creating edges (default: 0.7)
    max_connections : int, optional
        Maximum number of connections per node (default: 10)
    export_path : str, optional
        Base path for exported files (default: "network")
    mz_tolerance : float, optional
        Tolerance for merging similar m/z values (default: 1e-6)

    Returns
    -------
    G : networkx.Graph
        The created network graph
    cluster_labels : numpy.ndarray
        Cluster assignments for each spectrum
    connected_components : list
        List of connected components (subnetworks)
    precursor_mz_to_index : dict
        Mapping from precursor m/z to spectrum index

    Notes
    -----
    This function performs the following steps:
    1. Merges duplicate m/z values
    2. Extracts spectrum annotations
    3. Performs hierarchical clustering
    4. Creates network with limited connections per node
    5. Visualizes the network and subnetworks
    6. Exports to CSV and XGMML formats

    Output Files:
    - {export_path}_nodes.csv: Node attributes table
    - {export_path}_edges.csv: Edge list with attributes
    - {export_path}.xgmml: Cytoscape-compatible network file

    Examples
    --------
    >>> G, labels, components, mz_to_idx = create_similarity_network(
    ...     scores_array, spectrums,
    ...     distance_threshold=0.3,
    ...     similarity_cutoff=0.7,
    ...     max_connections=10,
    ...     export_path="results/network"
    ... )
    去重: 1000 -> 856 谱图
    网络统计: 425 节点, 1847 边
    发现 45 个子网络, 其中 8 个大型子网络
    """

    # 合并相似m/z值
    mz_to_merged, unique_spectrums, unique_to_original_indices = merge_duplicate_mz(spectrums, mz_tolerance)
    print(f"去重后谱图数量: {len(spectrums)} -> {len(unique_spectrums)}")

    # 提取谱图注释信息
    spectrum_annotations = extract_spectrum_annotations(spectrums, mz_to_merged)

    # 创建索引映射
    precursor_mz_to_index = {}
    index_to_precursor_mz = {}
    valid_indices = []

    for i, spectrum in enumerate(unique_spectrums):
        mz = float(spectrum.get('precursor_mz'))
        if mz is not None:
            merged_mz = mz_to_merged.get(mz, mz)
            precursor_mz_to_index[merged_mz] = i
            index_to_precursor_mz[i] = merged_mz
            valid_indices.append(i)

    print(f"有效索引数量: {len(valid_indices)} / {len(unique_spectrums)}")

    # 创建新的相似度矩阵（使用正确的索引映射）
    unique_scores = np.zeros((len(unique_spectrums), len(unique_spectrums)))
    for i in valid_indices:
        for j in valid_indices:
            # 获取原始索引（取每组的第一个代表）
            orig_i = unique_to_original_indices[i][0]
            orig_j = unique_to_original_indices[j][0]
            if orig_i < len(scores_array) and orig_j < len(scores_array[0]):
                unique_scores[i][j] = scores_array[orig_i][orig_j]

    # 层次聚类
    distances = 1 - unique_scores

    # 处理边界情况：样本数量不足
    if len(unique_spectrums) < 2:
        # 如果只有0或1个样本，不进行聚类
        cluster_labels = np.array([0] if len(unique_spectrums) == 1 else [])
        print(f"警告: 样本数量不足 ({len(unique_spectrums)})，跳过聚类步骤")
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage='complete')
        cluster_labels = clustering.fit_predict(distances)

        # 绘制聚类树状图
        plt.figure(figsize=(12, 6))
        Z = linkage(distances, method='complete')
        dendrogram(Z, truncate_mode='lastp', p=30)
        plt.title('层次聚类树状图')
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

    # 添加边（限制连接数）
    all_edges = []
    for i in range(len(unique_spectrums)):
        if i not in index_to_precursor_mz:
            continue
        for j in range(i + 1, len(unique_spectrums)):
            if j not in index_to_precursor_mz:
                continue
            similarity = unique_scores[i][j]
            if similarity >= similarity_cutoff:
                all_edges.append((index_to_precursor_mz[i], index_to_precursor_mz[j], similarity))

    all_edges.sort(key=lambda x: x[2], reverse=True)

    for source, target, similarity in all_edges:
        if G.degree(source) < max_connections and G.degree(target) < max_connections:
            G.add_edge(source, target, weight=similarity)

    # 网络可视化
    if G.number_of_nodes() > 0:
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()))

        node_colors = [cluster_labels[precursor_mz_to_index[node]] for node in G.nodes()]
        labels = {node: f"{float(node):.2f}" for node in G.nodes()}

        nx.draw(G, pos, node_color=node_colors, cmap=plt.cm.viridis,
                node_size=100, width=0.5, with_labels=True, labels=labels, font_size=8)
        plt.title(f'分子相似度网络 (最大连接数: {max_connections})')
        plt.show()

        # 子网络分析
        subnetworks = list(nx.connected_components(G))
        large_subnets = [s for s in subnetworks if len(s) > 2]

        print(f"网络统计: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        print(f"发现 {len(subnetworks)} 个子网络, 其中 {len(large_subnets)} 个大型子网络")

        # 显示前3个大型子网络
        for i, subnet_nodes in enumerate(large_subnets[:3]):
            subnet = G.subgraph(subnet_nodes).copy()
            plt.figure(figsize=(8, 6))
            pos_subnet = nx.spring_layout(subnet)

            node_colors_subnet = [cluster_labels[precursor_mz_to_index[node]]
                                for node in subnet.nodes()]
            subnet_labels = {node: f"{float(node):.2f}" for node in subnet.nodes()}

            nx.draw(subnet, pos_subnet, node_color=node_colors_subnet,
                   cmap=plt.cm.viridis, node_size=200, width=1,
                   with_labels=True, labels=subnet_labels, font_size=10)
            plt.title(f'子网络 {i+1} ({len(subnet_nodes)} 节点)')
            plt.show()

    # 导出数据
    nodes_data = []
    for node in G.nodes():
        node_data = {
            'id': str(node),
            'mz': float(node),
            'degree': G.degree(node),
            'cluster': cluster_labels[precursor_mz_to_index[node]]
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
    pd.DataFrame(nodes_data).to_csv(f"{export_path}_nodes.csv", index=False)
    pd.DataFrame(edges_data).to_csv(f"{export_path}_edges.csv", index=False)

    # 创建Cytoscape XGMML文件
    xgmml_file = create_cytoscape_xgmml(G, cluster_labels, precursor_mz_to_index,
                                        spectrum_annotations, f"{export_path}.xgmml")

    print(f"数据已导出:")
    print(f"  - CSV节点文件: {export_path}_nodes.csv")
    print(f"  - CSV边文件: {export_path}_edges.csv")
    print(f"  - Cytoscape文件: {export_path}.xgmml")

    return G, cluster_labels, list(nx.connected_components(G)), precursor_mz_to_index
