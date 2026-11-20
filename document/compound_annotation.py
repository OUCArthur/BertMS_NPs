#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
化合物注释和类型预测模块

功能：
1. 基于质谱相似度搜索数据库化合物
2. 注释化合物名称和SMILES
3. 预测化合物类型/类别
4. 集成到分子网络分析流程

Created by: Claude
Date: 2024-09-14
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs import FingerprintSimilarity
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CompoundInfo:
    """化合物信息数据类"""
    compound_id: str
    name: str
    smiles: str
    inchi: Optional[str] = None
    inchikey: Optional[str] = None
    formula: str = ""
    exact_mass: float = 0.0
    compound_class: str = "Unknown"
    similarity_score: float = 0.0
    database_source: str = ""
    peaks: Optional[List[Dict]] = field(default=None)  # MSP质谱峰数据
    precursor_mz: Optional[float] = None  # MSP前体离子质量
    metadata: Optional[Dict] = field(default=None)  # 其他MSP元数据


class CompoundDatabase:
    """化合物数据库管理类"""

    def __init__(self, database_path: Optional[str] = None):
        """
        初始化化合物数据库

        Args:
            database_path: 本地数据库文件路径
        """
        self.compounds = {}  # 存储化合物信息
        self.mz_index = defaultdict(list)  # m/z值索引
        self.smiles_index = {}  # SMILES索引
        self.database_path = database_path

        if database_path and os.path.exists(database_path):
            self.load_database(database_path)

    def load_database(self, file_path: str):
        """从文件加载化合物数据库"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif file_path.endswith('.msp'):
                # 加载MSP格式文件
                compounds_list = self._parse_msp_file(file_path)
                if compounds_list:
                    df = pd.DataFrame(compounds_list)
                else:
                    logging.warning(f"未能从MSP文件中解析到化合物: {file_path}")
                    return
            else:
                logging.warning(f"不支持的文件格式: {file_path}")
                return

            self._process_dataframe(df)
            logging.info(f"成功加载 {len(self.compounds)} 个化合物")

        except Exception as e:
            logging.error(f"加载数据库失败: {e}")

    def _parse_msp_file(self, file_path: str) -> List[Dict[str, Any]]:
        """解析MSP格式文件"""
        compounds = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_compound = {}
                peaks = []
                reading_peaks = False
                compound_counter = 1

                for line in f:
                    line = line.strip()

                    # 空行表示一个化合物记录的结束
                    if not line:
                        if current_compound:
                            # 如果没有ID，自动生成
                            if 'compound_id' not in current_compound:
                                current_compound['compound_id'] = f"MSP_{compound_counter:05d}"
                                compound_counter += 1

                            # 存储峰列表
                            if peaks:
                                current_compound['peaks'] = peaks

                            compounds.append(current_compound)
                            current_compound = {}
                            peaks = []
                            reading_peaks = False
                        continue

                    # 检查是否是键值对
                    if ':' in line and not reading_peaks:
                        key, value = line.split(':', 1)
                        key = key.strip().upper()
                        value = value.strip()

                        # 映射MSP字段到我们的数据结构
                        if key == 'NAME':
                            current_compound['name'] = value
                        elif key == 'PRECURSORMZ' or key == 'PRECURSOR_MZ':
                            current_compound['precursor_mz'] = float(value)
                        elif key == 'MW' or key == 'MOLECULARWEIGHT' or key == 'MOLECULAR_WEIGHT':
                            current_compound['exact_mass'] = float(value)
                        elif key == 'FORMULA' or key == 'MOLECULARFORMULA' or key == 'MOLECULAR_FORMULA':
                            current_compound['formula'] = value
                        elif key == 'INCHIKEY' or key == 'INCHI_KEY':
                            current_compound['inchikey'] = value
                        elif key == 'INCHI':
                            current_compound['inchi'] = value
                        elif key == 'SMILES':
                            current_compound['smiles'] = value
                        elif key == 'ID' or key == 'DB#' or key == 'ACCESSION':
                            current_compound['compound_id'] = value
                        elif key == 'COMPOUND_CLASS' or key == 'CLASS' or key == 'COMPOUNDCLASS':
                            current_compound['compound_class'] = value
                        elif key == 'NUM PEAKS' or key == 'NUMPEAKS' or key == 'NUM_PEAKS':
                            reading_peaks = True
                            continue
                        elif key == 'SOURCE' or key == 'DATABASE':
                            current_compound['database_source'] = value
                        elif key == 'IONMODE' or key == 'ION_MODE':
                            current_compound['ion_mode'] = value
                        elif key == 'COLLISIONENERGY' or key == 'COLLISION_ENERGY':
                            current_compound['collision_energy'] = value
                        # 存储其他字段作为元数据
                        else:
                            if 'metadata' not in current_compound:
                                current_compound['metadata'] = {}
                            current_compound['metadata'][key] = value

                    # 读取质谱峰数据
                    elif reading_peaks:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                mz = float(parts[0])
                                intensity = float(parts[1])
                                peaks.append({'mz': mz, 'intensity': intensity})
                            except ValueError:
                                continue

                # 处理最后一个化合物（如果文件没有以空行结束）
                if current_compound:
                    if 'compound_id' not in current_compound:
                        current_compound['compound_id'] = f"MSP_{compound_counter:05d}"
                    if peaks:
                        current_compound['peaks'] = peaks
                    compounds.append(current_compound)

                logging.info(f"从MSP文件中解析了 {len(compounds)} 个化合物")

                # 如果某些化合物缺少SMILES，尝试生成或设置为空字符串
                for compound in compounds:
                    if 'smiles' not in compound or not compound['smiles']:
                        # 如果有InChI，可以尝试转换（需要额外的库）
                        # 这里暂时设置为空字符串或占位符
                        compound['smiles'] = ''

                    # 确保必要字段存在
                    if 'name' not in compound:
                        compound['name'] = compound.get('compound_id', 'Unknown')

                    # 如果有precursor_mz但没有exact_mass，使用precursor_mz
                    if 'precursor_mz' in compound and 'exact_mass' not in compound:
                        # 假设是[M+H]+，减去质子质量
                        compound['exact_mass'] = compound['precursor_mz'] - 1.007276

                return compounds

        except Exception as e:
            logging.error(f"解析MSP文件时出错: {e}")
            return []

    def _process_dataframe(self, df: pd.DataFrame):
        """处理数据框并建立索引"""
        # 对于MSP格式，SMILES可能不是必需的
        required_columns = ['compound_id', 'name']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logging.error(f"数据库缺少必要列: {missing_columns}")
            return

        for _, row in df.iterrows():
            try:
                compound_id = str(row['compound_id'])
                name = str(row.get('name', ''))
                smiles = str(row.get('smiles', ''))

                # 如果有SMILES，验证其有效性
                mol = None
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)

                # 如果有分子式但没有SMILES，尝试使用分子式信息
                if mol is None and 'formula' in row and row['formula']:
                    # 对于没有SMILES的化合物，我们仍然可以存储其他信息
                    exact_mass = row.get('exact_mass', 0.0)
                    formula = row.get('formula', '')
                elif mol:
                    # 从SMILES计算分子量和分子式
                    exact_mass = Descriptors.ExactMolWt(mol)
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                else:
                    # 如果既没有SMILES也没有分子式，使用已有的数据
                    exact_mass = row.get('exact_mass', row.get('precursor_mz', 0.0))
                    formula = row.get('formula', '')

                compound_info = CompoundInfo(
                    compound_id=compound_id,
                    name=name,
                    smiles=smiles,
                    inchi=row.get('inchi', ''),
                    inchikey=row.get('inchikey', ''),
                    formula=formula,
                    exact_mass=exact_mass,
                    compound_class=row.get('compound_class', 'Unknown'),
                    database_source=row.get('database_source', 'MSP' if 'peaks' in row else 'Local')
                )

                # 存储额外的MSP特有信息
                if 'peaks' in row:
                    compound_info.peaks = row['peaks']
                if 'precursor_mz' in row:
                    compound_info.precursor_mz = row['precursor_mz']
                if 'metadata' in row:
                    compound_info.metadata = row['metadata']

                self.compounds[compound_id] = compound_info

                # 建立m/z索引
                if 'precursor_mz' in row and row['precursor_mz']:
                    # 如果有precursor_mz，直接使用它
                    precursor_mz = float(row['precursor_mz'])
                    self.mz_index[round(precursor_mz, 4)].append(compound_id)
                elif exact_mass > 0:
                    # 否则使用计算的质量建立索引 (考虑质子化)
                    mz_protonated = exact_mass + 1.007276  # [M+H]+
                    mz_deprotonated = exact_mass - 1.007276  # [M-H]-

                    self.mz_index[round(mz_protonated, 4)].append(compound_id)
                    self.mz_index[round(mz_deprotonated, 4)].append(compound_id)
                    self.mz_index[round(exact_mass, 4)].append(compound_id)

                # 建立SMILES索引（仅当SMILES存在时）
                if smiles:
                    self.smiles_index[smiles] = compound_id

            except Exception as e:
                logging.warning(f"处理化合物 {row.get('compound_id', 'unknown')} 时出错: {e}")


class CompoundClassifier:
    """化合物类型预测器"""

    def __init__(self):
        """初始化化合物分类器"""
        # 基于分子描述符的简单规则分类器
        # 实际应用中可以使用机器学习模型
        self.classification_rules = {
            'Lipid': lambda mol: self._is_lipid(mol),
            'Steroid': lambda mol: self._is_steroid(mol),
            'Alkaloid': lambda mol: self._is_alkaloid(mol),
            'Flavonoid': lambda mol: self._is_flavonoid(mol),
            'Phenolic': lambda mol: self._is_phenolic(mol),
            'Terpenoid': lambda mol: self._is_terpenoid(mol),
            'Amino Acid': lambda mol: self._is_amino_acid(mol),
            'Nucleotide': lambda mol: self._is_nucleotide(mol),
            'Sugar': lambda mol: self._is_sugar(mol),
            'Organic Acid': lambda mol: self._is_organic_acid(mol),
        }

    def predict_compound_class(self, smiles: str) -> str:
        """预测化合物类型"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "Unknown"

            # 按优先级检查各类型
            for compound_class, rule_func in self.classification_rules.items():
                if rule_func(mol):
                    return compound_class

            return "Other Organic"

        except Exception as e:
            logging.warning(f"预测化合物类型时出错: {e}")
            return "Unknown"

    def _is_lipid(self, mol) -> bool:
        """检查是否为脂类"""
        # 长链脂肪酸特征
        carbon_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        oxygen_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'])

        # 脂类通常有长碳链，C/O比例较高
        return carbon_count >= 12 and carbon_count / max(oxygen_count, 1) > 3

    def _is_steroid(self, mol) -> bool:
        """检查是否为类固醇"""
        # 类固醇的基本环状结构特征
        ring_info = mol.GetRingInfo()
        num_rings = len(ring_info.AtomRings())
        carbon_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])

        # 类固醇通常有3-4个环，碳原子数在17-30之间
        return num_rings >= 3 and 17 <= carbon_count <= 35

    def _is_alkaloid(self, mol) -> bool:
        """检查是否为生物碱"""
        # 生物碱通常含氮，且有环状结构
        nitrogen_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'])
        ring_info = mol.GetRingInfo()
        num_rings = len(ring_info.AtomRings())

        return nitrogen_count >= 1 and num_rings >= 1

    def _is_flavonoid(self, mol) -> bool:
        """检查是否为黄酮类"""
        # 黄酮类化合物的苯环和氧原子特征
        oxygen_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'])
        carbon_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        ring_info = mol.GetRingInfo()
        num_rings = len(ring_info.AtomRings())

        # 黄酮类通常有2-3个苯环，多个氧原子
        return num_rings >= 2 and oxygen_count >= 3 and 15 <= carbon_count <= 25

    def _is_phenolic(self, mol) -> bool:
        """检查是否为酚类"""
        # 查找苯环上的羟基
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() == 1:  # OH基团
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'C' and neighbor.GetIsAromatic():
                        return True
        return False

    def _is_terpenoid(self, mol) -> bool:
        """检查是否为萜类"""
        carbon_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        # 萜类的碳原子数通常是5的倍数
        return carbon_count % 5 == 0 and carbon_count >= 10

    def _is_amino_acid(self, mol) -> bool:
        """检查是否为氨基酸"""
        nitrogen_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'])
        oxygen_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'])
        carbon_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])

        # 氨基酸基本特征：含N和O，碳原子数较少
        return nitrogen_count >= 1 and oxygen_count >= 2 and 2 <= carbon_count <= 20

    def _is_nucleotide(self, mol) -> bool:
        """检查是否为核苷酸"""
        nitrogen_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'N'])
        phosphorus_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'P'])

        return nitrogen_count >= 2 and phosphorus_count >= 1

    def _is_sugar(self, mol) -> bool:
        """检查是否为糖类"""
        carbon_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        oxygen_count = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'O'])

        # 糖类的C:O比例通常接近1:1
        return 4 <= carbon_count <= 12 and abs(carbon_count - oxygen_count) <= 2

    def _is_organic_acid(self, mol) -> bool:
        """检查是否为有机酸"""
        # 查找羧基(-COOH)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                neighbors = list(atom.GetNeighbors())
                if len(neighbors) >= 2:
                    oxygen_neighbors = [n for n in neighbors if n.GetSymbol() == 'O']
                    if len(oxygen_neighbors) >= 2:
                        return True
        return False


class CompoundAnnotator:
    """化合物注释器主类"""

    def __init__(self, database_path: Optional[str] = None, mz_tolerance: float = 0.01):
        """
        初始化化合物注释器

        Args:
            database_path: 化合物数据库文件路径
            mz_tolerance: m/z值匹配容差
        """
        self.database = CompoundDatabase(database_path)
        self.classifier = CompoundClassifier()
        self.mz_tolerance = mz_tolerance

    def search_compounds_by_mz(self, target_mz: float, top_k: int = 5) -> List[CompoundInfo]:
        """
        根据m/z值搜索化合物

        Args:
            target_mz: 目标m/z值
            top_k: 返回前k个匹配结果

        Returns:
            匹配的化合物信息列表
        """
        candidates = []

        # 在容差范围内搜索
        for mz_key, compound_ids in self.database.mz_index.items():
            if abs(mz_key - target_mz) <= self.mz_tolerance:
                for compound_id in compound_ids:
                    compound_info = self.database.compounds[compound_id]
                    # 计算m/z匹配分数
                    mz_similarity = 1 - (abs(mz_key - target_mz) / self.mz_tolerance)
                    compound_info.similarity_score = mz_similarity
                    candidates.append(compound_info)

        # 按相似度排序并返回前k个
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        return candidates[:top_k]

    def annotate_spectrum(self, spectrum, top_k: int = 3) -> Dict[str, Any]:
        """
        为质谱图注释化合物信息

        Args:
            spectrum: 质谱图对象
            top_k: 返回前k个注释结果

        Returns:
            注释结果字典
        """
        try:
            # 获取前体离子m/z值
            precursor_mz = float(spectrum.get('precursor_mz'))

            # 搜索匹配的化合物
            candidates = self.search_compounds_by_mz(precursor_mz, top_k)

            if not candidates:
                return {
                    'precursor_mz': precursor_mz,
                    'best_match': None,
                    'all_candidates': [],
                    'predicted_class': 'Unknown',
                    'annotation_confidence': 0.0
                }

            # 选择最佳匹配
            best_match = candidates[0]

            # 预测化合物类型（使用最佳匹配的SMILES）
            predicted_class = self.classifier.predict_compound_class(best_match.smiles)

            return {
                'precursor_mz': precursor_mz,
                'best_match': {
                    'compound_id': best_match.compound_id,
                    'name': best_match.name,
                    'smiles': best_match.smiles,
                    'formula': best_match.formula,
                    'exact_mass': best_match.exact_mass,
                    'similarity_score': best_match.similarity_score,
                    'database_source': best_match.database_source
                },
                'all_candidates': [
                    {
                        'compound_id': c.compound_id,
                        'name': c.name,
                        'smiles': c.smiles,
                        'similarity_score': c.similarity_score
                    } for c in candidates
                ],
                'predicted_class': predicted_class,
                'annotation_confidence': best_match.similarity_score
            }

        except Exception as e:
            logging.error(f"注释质谱图时出错: {e}")
            return {
                'precursor_mz': None,
                'best_match': None,
                'all_candidates': [],
                'predicted_class': 'Unknown',
                'annotation_confidence': 0.0,
                'error': str(e)
            }

    def annotate_cluster(self, spectrums: List, cluster_id: int) -> Dict[str, Any]:
        """
        为聚类注释化合物信息

        Args:
            spectrums: 聚类中的质谱图列表
            cluster_id: 聚类ID

        Returns:
            聚类注释结果
        """
        annotations = []
        compound_classes = []

        for spectrum in spectrums:
            annotation = self.annotate_spectrum(spectrum)
            annotations.append(annotation)

            if annotation['predicted_class'] != 'Unknown':
                compound_classes.append(annotation['predicted_class'])

        # 统计聚类中的化合物类型
        class_counts = Counter(compound_classes)
        dominant_class = class_counts.most_common(1)[0][0] if class_counts else 'Unknown'

        # 选择最佳代表性注释
        valid_annotations = [a for a in annotations if a['best_match'] is not None]
        representative_annotation = None

        if valid_annotations:
            # 选择置信度最高的注释作为代表
            representative_annotation = max(valid_annotations,
                                          key=lambda x: x['annotation_confidence'])

        return {
            'cluster_id': cluster_id,
            'cluster_size': len(spectrums),
            'dominant_class': dominant_class,
            'class_distribution': dict(class_counts),
            'representative_annotation': representative_annotation,
            'all_annotations': annotations,
            'average_confidence': np.mean([a['annotation_confidence'] for a in annotations])
        }


def create_sample_database():
    """创建示例化合物数据库"""
    sample_compounds = [
        {
            'compound_id': 'COMP_001',
            'name': 'Caffeine',
            'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'compound_class': 'Alkaloid',
            'database_source': 'Sample'
        },
        {
            'compound_id': 'COMP_002',
            'name': 'Glucose',
            'smiles': 'C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O',
            'compound_class': 'Sugar',
            'database_source': 'Sample'
        },
        {
            'compound_id': 'COMP_003',
            'name': 'Cholesterol',
            'smiles': 'C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C',
            'compound_class': 'Steroid',
            'database_source': 'Sample'
        }
    ]

    df = pd.DataFrame(sample_compounds)
    return df


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建示例数据库
    print("创建示例化合物数据库...")
    sample_db = create_sample_database()
    sample_db.to_csv('sample_compound_database.csv', index=False)

    # 初始化注释器
    annotator = CompoundAnnotator('sample_compound_database.csv')

    # 测试搜索功能
    print("测试m/z搜索功能...")
    results = annotator.search_compounds_by_mz(194.08, top_k=3)
    for result in results:
        print(f"化合物: {result.name}, 相似度: {result.similarity_score:.3f}")

    print("化合物注释模块创建完成！")