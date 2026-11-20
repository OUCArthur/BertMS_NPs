#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
化合物重分类脚本

使用RDKit和规则系统对GNPS数据中的化合物进行更细粒度的分类

支持的类别：
1. 多肽/环肽 (Peptides/Cyclic Peptides)
2. 糖类 (Carbohydrates)
3. 生物碱 (Alkaloids)
4. 脂肪酸/脂质 (Fatty Acids/Lipids)
5. 萜类 (Terpenoids)
6. 类固醇 (Steroids)
7. 黄酮类 (Flavonoids)
8. 聚酮化合物 (Polyketides)
9. 磷脂 (Phospholipids)
10. 胆汁酸 (Bile Acids)
11. 吲哚衍生物 (Indole Derivatives)
12. 其他 (Others)
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import tqdm, use fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc='', **kwargs):
        """Fallback progress bar"""
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total and (i % max(1, total // 20) == 0 or i == total - 1):
                print(f"\r{desc}: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end='', flush=True)
            yield item
        print()  # New line after progress

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Fragments, Lipinski
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Classification will be based on compound names only.")


class CompoundClassifier:
    """化合物分类器"""

    def __init__(self):
        """初始化分类器"""
        self.category_keywords = {
            'Peptides': [
                'peptide', 'dipeptide', 'tripeptide', 'cyclo', 'gly-', 'ala-', 'val-',
                'leu-', 'ile-', 'phe-', 'trp-', 'tyr-', 'ser-', 'thr-', 'cys-', 'met-',
                'asn-', 'gln-', 'asp-', 'glu-', 'lys-', 'arg-', 'his-', 'pro-',
                'surfactin', 'gramicidin', 'bacitracin', 'polymyxin'
            ],
            'Carbohydrates': [
                'sugar', 'glucose', 'fructose', 'galactose', 'mannose', 'ribose',
                'xylose', 'arabinose', 'xylitol', 'sorbitol', 'mannitol', 'glyco',
                'oligo', 'poly', 'saccharide', 'cellulose', 'starch', 'glycan'
            ],
            'Alkaloids': [
                'alkaloid', 'cocaine', 'morphine', 'codeine', 'caffeine', 'nicotine',
                'atropine', 'quinine', 'strychnine', 'colchicine', 'brucine',
                'berberine', 'sanguinarine', 'piperine', '-ine', 'tropane'
            ],
            'Fatty Acids/Lipids': [
                'fatty acid', 'palmitic', 'stearic', 'oleic', 'linoleic', 'linolenic',
                'arachidonic', 'docosahexaenoic', 'eicosapentaenoic', 'myristate',
                'laurate', 'caprate', 'caproate', 'butyrate', 'acetate', 'lipid',
                'triglyceride', 'diglyceride', 'monoglyceride', 'c8:', 'c9:', 'c10:',
                'c11:', 'c12:', 'c13:', 'c14:', 'c15:', 'c16:', 'c17:', 'c18:', 'c20:'
            ],
            'Terpenoids': [
                'terpene', 'terpenoid', 'monoterpene', 'sesquiterpene', 'diterpene',
                'triterpene', 'carotenoid', 'menthol', 'limonene', 'pinene', 'camphor',
                'geraniol', 'farnesol', 'squalene', 'abietic', 'cembratriene'
            ],
            'Steroids': [
                'steroid', 'testosterone', 'estrogen', 'progesterone', 'cortisol',
                'cholesterol', 'ergosterol', 'stigmasterol', 'calusterone', 'furazabol',
                'mesterolone', 'metenolone', 'stanozolol', '-sterone', '-sterol'
            ],
            'Flavonoids': [
                'flavonoid', 'flavone', 'flavonol', 'flavanone', 'isoflavone',
                'anthocyanin', 'quercetin', 'kaempferol', 'luteolin', 'apigenin',
                'catechin', 'epicatechin', 'naringenin', 'hesperidin', 'rutin'
            ],
            'Polyketides': [
                'polyketide', 'macrolide', 'erythromycin', 'azithromycin', 'rapamycin',
                'tetracycline', 'doxorubicin', 'amphotericin', 'lovastatin', 'aflatoxin',
                'zearalenone', 'ochratoxin', 'curvularin', 'destruxin'
            ],
            'Phospholipids': [
                'phospholipid', 'phosphatidyl', 'phosphatidylcholine', 'phosphatidylethanolamine',
                'phosphatidylserine', 'phosphatidylinositol', 'lecithin', 'pc(', 'pe(',
                'ps(', 'pi(', 'plpe', 'plpc'
            ],
            'Bile Acids': [
                'bile acid', 'cholic', 'chenodeoxycholic', 'deoxycholic', 'lithocholic',
                'ursodeoxycholic', '-cdca', '-dca', '-hdca', '-udca', '-ca', 'cholate',
                'ornithine-', 'taurine-', 'glycine-'
            ],
            'Indole Derivatives': [
                'indole', 'indoline', 'oxindole', 'tryptophan', 'tryptamine', 'serotonin',
                'melatonin', 'indoleacetic', 'skatole', 'carbazole', 'isatin'
            ],
            'Antibiotics': [
                'antibiotic', 'penicillin', 'cephalosporin', 'vancomycin', 'streptomycin',
                'gentamicin', 'kanamycin', 'chloramphenicol', 'rifampicin'
            ]
        }

        # SMARTS patterns for structure-based classification
        if RDKIT_AVAILABLE:
            self.smarts_patterns = {
                'Peptides': '[NX3][CX3](=[OX1])[CX4][NX3]',  # Peptide bond pattern
                'Carbohydrates': '[CH]([OH])[CH]([OH])[CH]([OH])',  # Sugar pattern
                'Indole Derivatives': 'c1ccc2c(c1)[nH]cc2',  # Indole core
                'Steroids': '[#6]~1~[#6]~[#6]~[#6]~2~[#6]~[#6]~[#6]~3~[#6]~[#6]~[#6]~4~[#6]~[#6]~[#6]~[#6]~[#6]~4~[#6]~[#6]~3~[#6]~[#6]~2~[#6]~1',  # Steroid skeleton
                'Phospholipids': '[P](=[O])([O-])([O])',  # Phosphate group
            }
        else:
            self.smarts_patterns = {}

    def classify_by_name(self, compound_name: str) -> Tuple[str, float]:
        """
        根据化合物名称进行分类

        Args:
            compound_name: 化合物名称

        Returns:
            (category, confidence): 类别和置信度
        """
        if not compound_name or compound_name.strip() in ['', 'N/A', 'Unknown']:
            return 'Others', 0.0

        compound_name_lower = compound_name.lower()

        # 检查每个类别的关键词
        scores = {}
        for category, keywords in self.category_keywords.items():
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in compound_name_lower:
                    score += 1
                    matched_keywords.append(keyword)

            if score > 0:
                # 根据匹配的关键词数量和长度计算置信度
                confidence = min(score * 0.3 + len(max(matched_keywords, key=len)) * 0.02, 1.0)
                scores[category] = confidence

        if not scores:
            return 'Others', 0.0

        # 返回最高分的类别
        best_category = max(scores.items(), key=lambda x: x[1])
        return best_category[0], best_category[1]

    def classify_by_smiles(self, smiles: str) -> Tuple[Optional[str], float]:
        """
        根据SMILES结构进行分类

        Args:
            smiles: SMILES字符串

        Returns:
            (category, confidence): 类别和置信度
        """
        if not RDKIT_AVAILABLE:
            return None, 0.0

        if not smiles or smiles.strip() in ['', 'N/A', 'Unknown']:
            return None, 0.0

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, 0.0

            # 使用SMARTS模式匹配
            matches = {}
            for category, pattern in self.smarts_patterns.items():
                patt = Chem.MolFromSmarts(pattern)
                if patt and mol.HasSubstructMatch(patt):
                    matches[category] = 0.8  # 结构匹配的置信度

            # 基于分子描述符的额外规则
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            num_rings = Descriptors.RingCount(mol)
            num_aromatic_rings = Descriptors.NumAromaticRings(mol)

            # 脂肪酸规则
            if 200 <= mw <= 400 and logp > 3 and num_rings <= 1:
                if 'Fatty Acids/Lipids' not in matches:
                    matches['Fatty Acids/Lipids'] = 0.6

            # 萜类规则 (通常是C5的倍数)
            num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
            if num_carbons % 5 == 0 and 10 <= num_carbons <= 30 and num_rings >= 1:
                if 'Terpenoids' not in matches:
                    matches['Terpenoids'] = 0.5

            # 类黄酮规则
            if num_aromatic_rings >= 2 and 200 <= mw <= 400:
                if 'Flavonoids' not in matches:
                    matches['Flavonoids'] = 0.5

            if not matches:
                return None, 0.0

            # 返回最高分的类别
            best_category = max(matches.items(), key=lambda x: x[1])
            return best_category[0], best_category[1]

        except Exception as e:
            print(f"Error processing SMILES {smiles[:50]}: {e}")
            return None, 0.0

    def classify_compound(self, compound_name: str, smiles: str) -> Tuple[str, float]:
        """
        综合分类（优先使用结构，然后使用名称）

        Args:
            compound_name: 化合物名称
            smiles: SMILES字符串

        Returns:
            (category, confidence): 类别和置信度
        """
        # 1. 先尝试基于结构分类
        category_struct, conf_struct = self.classify_by_smiles(smiles)

        # 2. 再尝试基于名称分类
        category_name, conf_name = self.classify_by_name(compound_name)

        # 3. 综合决策
        if category_struct and conf_struct > 0.6:
            # 结构分类置信度高，使用结构分类
            return category_struct, conf_struct
        elif category_struct and category_name == category_struct:
            # 结构和名称一致，提高置信度
            return category_struct, min(conf_struct + conf_name * 0.5, 1.0)
        elif conf_name > conf_struct if conf_struct else True:
            # 名称分类置信度更高
            return category_name, conf_name
        elif category_struct:
            return category_struct, conf_struct
        else:
            return category_name if category_name else 'Others', conf_name


def reclassify_gnps_data(input_path: str, output_path: str, min_confidence: float = 0.3):
    """
    对GNPS数据进行重分类

    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        min_confidence: 最小置信度阈值
    """
    print(f"\n{'='*60}")
    print("化合物重分类工具")
    print(f"{'='*60}\n")

    # 加载数据
    print(f"加载数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[OK] 加载了 {len(data)} 个化合物\n")

    # 初始化分类器
    classifier = CompoundClassifier()

    # 重分类
    print("开始重分类...")
    category_counts = Counter()
    low_confidence_count = 0

    for item in tqdm(data, desc="分类进度"):
        compound_name = item.get('compound_name', '')
        smiles = item.get('smiles', '')

        # 分类
        category, confidence = classifier.classify_compound(compound_name, smiles)

        # 如果置信度太低，标记为Others
        if confidence < min_confidence:
            category = 'Others'
            low_confidence_count += 1

        # 添加新字段
        item['detailed_class'] = category
        item['class_confidence'] = round(confidence, 4)

        category_counts[category] += 1

    # 保存结果
    print(f"\n保存重分类结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("[OK] 保存完成\n")

    # 统计报告
    print(f"{'='*60}")
    print("分类统计报告")
    print(f"{'='*60}\n")

    total = len(data)
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"{'类别':<30} {'数量':>8} {'占比':>8} {'可视化'}")
    print(f"{'-'*60}")
    for category, count in sorted_categories:
        percentage = count / total * 100
        bar_length = int(percentage / 2)
        bar = '#' * bar_length
        print(f"{category:<30} {count:>8} {percentage:>7.2f}% {bar}")

    print(f"\n低置信度(<{min_confidence})样本: {low_confidence_count} ({low_confidence_count/total*100:.2f}%)")
    print(f"\n总计: {total} 个化合物")
    print(f"{'='*60}\n")

    # 生成详细报告
    report_path = output_path.replace('.json', '_classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("化合物重分类报告\n")
        f.write("="*60 + "\n\n")

        f.write(f"输入文件: {input_path}\n")
        f.write(f"输出文件: {output_path}\n")
        f.write(f"总化合物数: {total}\n")
        f.write(f"分类数: {len(category_counts)}\n\n")

        f.write("类别分布:\n")
        f.write("-"*60 + "\n")
        for category, count in sorted_categories:
            percentage = count / total * 100
            f.write(f"{category:<30} {count:>8} ({percentage:>6.2f}%)\n")

        f.write("\n" + "="*60 + "\n")
        f.write("每个类别的示例化合物 (前10个):\n")
        f.write("="*60 + "\n\n")

        for category, _ in sorted_categories:
            f.write(f"\n{category}:\n")
            f.write("-"*40 + "\n")
            examples = [item for item in data if item['detailed_class'] == category][:10]
            for i, item in enumerate(examples, 1):
                name = item.get('compound_name', 'N/A')[:50]
                conf = item.get('class_confidence', 0)
                f.write(f"  {i:2d}. {name:<50} (置信度: {conf:.3f})\n")

    print(f"详细报告已保存到: {report_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='对GNPS数据进行化合物重分类')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入JSON文件路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出JSON文件路径')
    parser.add_argument('--min-confidence', '-c', type=float, default=0.3,
                        help='最小置信度阈值 (默认: 0.3)')

    args = parser.parse_args()

    # 执行重分类
    reclassify_gnps_data(args.input, args.output, args.min_confidence)


if __name__ == '__main__':
    # 示例用法
    print("\n" + "="*60)
    print("化合物重分类工具")
    print("="*60)
    print("\n使用方法:")
    print("\npython utils/reclassify_compounds.py \\")
    print("    --input data/ms/gnps_all_cleand_by_matchms_and_pubchem_lookups_1000.json \\")
    print("    --output data/ms/gnps_reclassified_1000.json \\")
    print("    --min-confidence 0.3")
    print("\n" + "="*60 + "\n")

    # 如果直接运行，使用默认参数
    input_file = "data/ms/gnps_all_cleand_by_matchms_and_pubchem_lookups_1000.json"
    output_file = "data/ms/gnps_reclassified_1000.json"

    if Path(input_file).exists():
        print(f"发现数据文件: {input_file}")
        print("开始自动重分类...\n")
        reclassify_gnps_data(input_file, output_file, min_confidence=0.3)
    else:
        print(f"数据文件不存在: {input_file}")
        print("请使用命令行参数指定输入文件")
