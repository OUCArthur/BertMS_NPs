"""
结构类别预测模块
使用机器学习方法预测化合物的结构类别

支持两种模式:
1. 基于质谱特征的预测（原有方法）
2. 基于SMILES分子指纹的预测（新方法，需要训练的模型）
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# 尝试导入RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class StructuralClassPredictor:
    """结构类型预测器

    支持两种预测模式:
    1. 基于质谱特征预测（默认，不需要额外模型）
    2. 基于SMILES分子结构预测（需要加载训练好的模型）
    """

    def __init__(self, model_path=None):
        """
        初始化结构类型预测器

        参数:
            model_path: 可选，训练好的SMILES分类模型路径
                       如果提供，将使用基于SMILES的预测
                       否则使用基于质谱特征的预测

        包含的结构类别：
        - Alkaloids: 生物碱
        - Amino acids: 氨基酸
        - Carbohydrates: 碳水化合物
        - Fatty acids: 脂肪酸
        - Flavonoids: 黄酮类
        - Nucleotides: 核苷酸
        - Peptides: 肽类
        - Phenolic compounds: 酚类化合物
        - Steroids: 甾体类
        - Terpenoids: 萜类
        - Unknown: 未知类别
        """
        # 基于质谱特征的预测器（原有功能）
        self.feature_extractor = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.structural_classes = [
            'Alkaloids', 'Amino acids', 'Carbohydrates', 'Fatty acids',
            'Flavonoids', 'Nucleotides', 'Peptides', 'Phenolic compounds',
            'Steroids', 'Terpenoids', 'Unknown'
        ]

        # 基于SMILES的预测器（新功能）
        self.smiles_model = None
        self.smiles_scaler = None
        self.smiles_label_encoder = None
        self.use_smiles_model = False

        # 深度学习模型（基于质谱特征）
        self.dl_predictor = None
        self.use_dl_model = False

        # 如果提供了模型路径，尝试加载模型
        if model_path is not None:
            # 检测模型类型
            if model_path.endswith('.pth'):
                # PyTorch深度学习模型
                self.load_dl_model(model_path)
            else:
                # SMILES分类模型
                self.load_smiles_model(model_path)

    def extract_spectral_features(self, spectrum):
        """
        从质谱中提取特征

        参数:
            spectrum: 质谱对象，需包含peaks属性

        返回:
            Dict: 特征字典，包含以下特征：
                - 基本统计特征: total_peaks, max_intensity, mean_intensity, std_intensity
                - m/z范围特征: mz_range, mean_mz, std_mz
                - 强度分布特征: top3_intensity_ratio
                - 峰密度特征: peak_density
                - 损失峰特征: loss_18, loss_28, loss_44, loss_60, loss_88
        """
        try:
            features = {}

            if not hasattr(spectrum, 'peaks') or len(spectrum.peaks.mz) == 0:
                return self._get_empty_features()

            mz_values = spectrum.peaks.mz
            intensities = spectrum.peaks.intensities

            # 基本统计特征
            features['total_peaks'] = len(mz_values)
            features['max_intensity'] = np.max(intensities)
            features['mean_intensity'] = np.mean(intensities)
            features['std_intensity'] = np.std(intensities)

            # m/z范围特征
            features['mz_range'] = np.max(mz_values) - np.min(mz_values)
            features['mean_mz'] = np.mean(mz_values)
            features['std_mz'] = np.std(mz_values)

            # 强度分布特征
            sorted_intensities = np.sort(intensities)[::-1]
            if len(sorted_intensities) >= 3:
                features['top3_intensity_ratio'] = np.sum(sorted_intensities[:3]) / np.sum(sorted_intensities)
            else:
                features['top3_intensity_ratio'] = 1.0

            # 峰密度特征
            if features['mz_range'] > 0:
                features['peak_density'] = features['total_peaks'] / features['mz_range']
            else:
                features['peak_density'] = 0

            # 损失峰特征（相对于母离子）
            precursor_mz = float(spectrum.get('precursor_mz', 0))
            if precursor_mz > 0:
                neutral_losses = precursor_mz - mz_values[mz_values < precursor_mz]
                common_losses = [18, 28, 44, 60, 88]  # H2O, CO, CO2, CH3COOH, etc.
                for loss in common_losses:
                    loss_count = np.sum(np.abs(neutral_losses - loss) < 0.1)
                    features[f'loss_{loss}'] = loss_count

            return features

        except Exception as e:
            print(f"特征提取出错: {e}")
            return self._get_empty_features()

    def _get_empty_features(self):
        """
        返回空特征向量

        返回:
            Dict: 所有特征值为0的特征字典
        """
        return {
            'total_peaks': 0, 'max_intensity': 0, 'mean_intensity': 0,
            'std_intensity': 0, 'mz_range': 0, 'mean_mz': 0, 'std_mz': 0,
            'top3_intensity_ratio': 0, 'peak_density': 0,
            'loss_18': 0, 'loss_28': 0, 'loss_44': 0, 'loss_60': 0, 'loss_88': 0
        }

    def create_training_data(self, spectrums, annotations):
        """
        创建训练数据（基于已知注释）

        参数:
            spectrums: 质谱对象列表
            annotations: 注释信息（当前版本未使用）

        返回:
            Tuple[np.ndarray, np.ndarray]: (特征矩阵, 标签向量)
        """
        training_features = []
        training_labels = []

        for spectrum in spectrums:
            features = self.extract_spectral_features(spectrum)
            feature_vector = list(features.values())

            # 根据化合物名称或SMILES推断结构类型
            compound_name = spectrum.get('compound_name', '').lower()
            smiles = spectrum.get('smiles', '')

            predicted_class = self._predict_class_from_metadata(compound_name, smiles)

            training_features.append(feature_vector)
            training_labels.append(predicted_class)

        return np.array(training_features), np.array(training_labels)

    def _predict_class_from_metadata(self, compound_name, smiles):
        """
        基于化合物名称和SMILES预测结构类型

        参数:
            compound_name: 化合物名称
            smiles: SMILES结构式

        返回:
            str: 预测的结构类别

        注意:
            使用基于关键词的简单规则进行分类
        """
        name = compound_name.lower()

        # 基于名称关键词的简单分类
        if any(word in name for word in ['alkaloid', 'ine', 'inine']):
            return 'Alkaloids'
        elif any(word in name for word in ['amino', 'glycine', 'alanine']):
            return 'Amino acids'
        elif any(word in name for word in ['glucose', 'sugar', 'saccharide']):
            return 'Carbohydrates'
        elif any(word in name for word in ['acid', 'fatty']):
            return 'Fatty acids'
        elif any(word in name for word in ['flavon', 'quercetin', 'rutin']):
            return 'Flavonoids'
        elif any(word in name for word in ['nucleotide', 'atp', 'gtp']):
            return 'Nucleotides'
        elif any(word in name for word in ['peptide', 'dipeptide']):
            return 'Peptides'
        elif any(word in name for word in ['phenol', 'catechin']):
            return 'Phenolic compounds'
        elif any(word in name for word in ['steroid', 'cholesterol']):
            return 'Steroids'
        elif any(word in name for word in ['terpene', 'menthol']):
            return 'Terpenoids'
        else:
            return 'Unknown'

    def train_model(self, spectrums, annotations=None):
        """
        训练结构类型预测模型

        参数:
            spectrums: 质谱对象列表
            annotations: 注释信息（可选）

        返回:
            bool: 训练是否成功
        """
        try:
            X, y = self.create_training_data(spectrums, annotations)

            if len(set(y)) < 2:
                print("训练数据类别太少，使用默认预测")
                self.is_trained = False
                return False

            self.classifier.fit(X, y)
            self.is_trained = True
            print(f"模型训练完成，包含 {len(set(y))} 个结构类型")
            return True

        except Exception as e:
            print(f"模型训练出错: {e}")
            self.is_trained = False
            return False

    def load_smiles_model(self, model_path):
        """
        加载基于SMILES的分类模型

        参数:
            model_path: 模型文件路径
        """
        try:
            if not os.path.exists(model_path):
                print(f"警告: 模型文件不存在: {model_path}")
                return False

            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.smiles_model = model_data['model']
            self.smiles_scaler = model_data['scaler']
            self.smiles_label_encoder = model_data['label_encoder']
            self.use_smiles_model = True

            print(f"✓ SMILES分类模型已加载")
            print(f"  模型类型: {model_data.get('model_type', 'unknown')}")
            print(f"  类别数: {len(self.smiles_label_encoder.classes_)}")
            print(f"  类别: {list(self.smiles_label_encoder.classes_)}")

            return True

        except Exception as e:
            print(f"加载SMILES模型失败: {e}")
            self.use_smiles_model = False
            return False

    def extract_smiles_features(self, smiles):
        """
        从SMILES提取分子特征（Morgan指纹+描述符）

        参数:
            smiles: SMILES字符串

        返回:
            np.ndarray: 特征向量，失败返回None
        """
        if not RDKIT_AVAILABLE:
            print("警告: RDKit未安装，无法提取SMILES特征")
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            features_list = []

            # 分子描述符
            descriptors = [
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.RingCount(mol),
                Descriptors.FractionCsp3(mol),
            ]
            features_list.append(np.array(descriptors, dtype=np.float32))

            # Morgan指纹
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            features_list.append(np.array(fp, dtype=np.float32))

            return np.concatenate(features_list)

        except Exception as e:
            print(f"提取SMILES特征失败: {e}")
            return None

    def predict_from_smiles(self, smiles):
        """
        基于SMILES预测结构类别

        参数:
            smiles: SMILES字符串

        返回:
            Tuple[str, float]: (预测的结构类别, 置信度)
        """
        if not self.use_smiles_model:
            return 'Unknown', 0.0

        try:
            # 提取特征
            features = self.extract_smiles_features(smiles)
            if features is None:
                return 'Unknown', 0.0

            # 处理异常值
            features = np.nan_to_num(features.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)

            # 标准化
            features_scaled = self.smiles_scaler.transform(features)

            # 预测
            pred_encoded = self.smiles_model.predict(features_scaled)[0]
            predicted_class = self.smiles_label_encoder.inverse_transform([pred_encoded])[0]

            # 获取置信度
            pred_proba = self.smiles_model.predict_proba(features_scaled)[0]
            confidence = pred_proba[pred_encoded]

            return predicted_class, confidence

        except Exception as e:
            print(f"SMILES预测失败: {e}")
            return 'Unknown', 0.0

    def load_dl_model(self, model_path):
        """
        加载深度学习模型（基于质谱特征）

        参数:
            model_path: 模型文件路径（.pth格式）
        """
        try:
            if not os.path.exists(model_path):
                print(f"警告: 模型文件不存在: {model_path}")
                return False

            # 导入深度学习预测器
            from .dl_classifier import DeepLearningClassifierPredictor

            self.dl_predictor = DeepLearningClassifierPredictor(model_path)
            self.use_dl_model = True

            print(f"✓ 深度学习分类模型已加载（基于质谱特征）")

            return True

        except Exception as e:
            print(f"加载深度学习模型失败: {e}")
            import traceback
            traceback.print_exc()
            self.use_dl_model = False
            return False

    def predict_structural_class(self, spectrum):
        """
        预测单个质谱的结构类型

        参数:
            spectrum: 质谱对象

        返回:
            Tuple[str, float]: (预测的结构类别, 置信度)
        """
        try:
            # 优先1：使用深度学习模型（基于质谱特征，不需要SMILES）
            if self.use_dl_model:
                try:
                    predicted_class, confidence = self.dl_predictor.predict(spectrum)
                    if predicted_class != 'Unknown' and confidence > 0.3:
                        return predicted_class, confidence
                except Exception as e:
                    print(f"深度学习预测失败，回退到其他方法: {e}")

            # 优先2：使用SMILES模型（如果可用且spectrum包含SMILES）
            if self.use_smiles_model:
                smiles = spectrum.get('smiles', '')
                if smiles:
                    predicted_class, confidence = self.predict_from_smiles(smiles)
                    if predicted_class != 'Unknown':
                        return predicted_class, confidence

            # 优先3：使用训练好的质谱特征模型
            if self.is_trained:
                features = self.extract_spectral_features(spectrum)
                feature_vector = np.array(list(features.values())).reshape(1, -1)

                prediction = self.classifier.predict(feature_vector)[0]
                probabilities = self.classifier.predict_proba(feature_vector)[0]
                confidence = np.max(probabilities)

                return prediction, confidence

            # 最后：基于元数据的规则预测
            compound_name = spectrum.get('compound_name', '')
            smiles = spectrum.get('smiles', '')
            predicted_class = self._predict_class_from_metadata(compound_name, smiles)
            return predicted_class, 0.5  # 默认置信度

        except Exception as e:
            print(f"结构类型预测出错: {e}")
            return 'Unknown', 0.0

    def save_model(self, filepath):
        """
        保存训练好的模型

        参数:
            filepath: 保存路径
        """
        try:
            model_data = {
                'classifier': self.classifier,
                'feature_extractor': self.feature_extractor,
                'is_trained': self.is_trained,
                'structural_classes': self.structural_classes
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"模型已保存到: {filepath}")
        except Exception as e:
            print(f"保存模型出错: {e}")

    def load_model(self, filepath):
        """
        加载训练好的模型

        参数:
            filepath: 模型文件路径
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.classifier = model_data['classifier']
            self.feature_extractor = model_data['feature_extractor']
            self.is_trained = model_data['is_trained']
            self.structural_classes = model_data['structural_classes']
            print(f"模型已从 {filepath} 加载")
        except Exception as e:
            print(f"加载模型出错: {e}")


if __name__ == '__main__':
    print("结构类别预测模块已加载")
    print(f"支持的结构类别: {', '.join(StructuralClassPredictor().structural_classes)}")
