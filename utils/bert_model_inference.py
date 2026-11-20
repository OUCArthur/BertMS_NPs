#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT分类器推理工具
用于在 Jupyter Notebook 中加载和使用训练好的BERT质谱分类模型

Created by: BertMS Project
Date: 2025-01-24
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))


class BertSpectrumClassifierInference:
    """BERT质谱分类器推理类"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化分类器

        Args:
            model_path: 模型文件路径 (.pth)
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)

        # 加载模型
        print(f"加载模型: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 提取配置
        self.model_config = checkpoint['model_config']
        self.label_mapping = checkpoint['label_mapping']
        self.performance = checkpoint.get('performance', {})
        self.training_config = checkpoint.get('config', {})

        # 重建分类器
        self.classifier = self._build_classifier()
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        # 标签映射
        self.label_to_idx = self.label_mapping['label_to_idx']
        self.idx_to_label = {int(k): v for k, v in self.label_mapping['idx_to_label'].items()}

        print(f"✓ 模型加载成功")
        print(f"  类别数量: {self.model_config['num_classes']}")
        print(f"  BERT向量维度: {self.model_config['hidden_size']}")
        print(f"  设备: {self.device}")
        if self.performance:
            print(f"  训练性能: {self.performance.get('best_val_acc', 'N/A'):.2f}% "
                  f"(Epoch {self.performance.get('best_epoch', 'N/A')})")

    def _build_classifier(self) -> nn.Module:
        """重建分类器"""
        hidden_size = self.model_config['hidden_size']
        classifier_hidden_dim = self.model_config['classifier_hidden_dim']
        num_classes = self.model_config['num_classes']

        classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.0),  # 推理时不需要dropout
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(classifier_hidden_dim // 2, num_classes)
        )

        return classifier

    @torch.no_grad()
    def predict_from_bert_vector(
        self,
        bert_vector: np.ndarray,
        return_probabilities: bool = False
    ) -> Dict:
        """
        从BERT向量预测类别

        Args:
            bert_vector: BERT特征向量 [hidden_size]
            return_probabilities: 是否返回所有类别的概率

        Returns:
            预测结果字典
        """
        # 转换为tensor
        if isinstance(bert_vector, np.ndarray):
            bert_vector = torch.FloatTensor(bert_vector)

        bert_vector = bert_vector.unsqueeze(0).to(self.device)  # [1, hidden_size]

        # 前向传播
        outputs = self.classifier(bert_vector)

        # 计算概率
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

        # 转换结果
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
        predicted_class = self.idx_to_label[predicted_idx]

        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_idx': predicted_idx
        }

        if return_probabilities:
            probs_dict = {
                self.idx_to_label[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
            result['probabilities'] = probs_dict

        return result

    @torch.no_grad()
    def predict_batch_from_bert_vectors(
        self,
        bert_vectors: np.ndarray,
        batch_size: int = 64,
        return_probabilities: bool = False
    ) -> List[Dict]:
        """
        批量从BERT向量预测

        Args:
            bert_vectors: BERT特征向量数组 [num_samples, hidden_size]
            batch_size: 批大小
            return_probabilities: 是否返回概率

        Returns:
            预测结果列表
        """
        results = []

        # 转换为tensor
        if isinstance(bert_vectors, np.ndarray):
            bert_vectors = torch.FloatTensor(bert_vectors)

        num_samples = bert_vectors.shape[0]

        for i in range(0, num_samples, batch_size):
            batch_vectors = bert_vectors[i:i+batch_size].to(self.device)

            # 前向传播
            outputs = self.classifier(batch_vectors)

            # 计算概率
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted_indices = probabilities.max(1)

            # 转换结果
            for j in range(len(batch_vectors)):
                confidence = confidences[j].item()
                predicted_idx = predicted_indices[j].item()
                predicted_class = self.idx_to_label[predicted_idx]

                result = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'class_idx': predicted_idx
                }

                if return_probabilities:
                    probs_dict = {
                        self.idx_to_label[k]: prob.item()
                        for k, prob in enumerate(probabilities[j])
                    }
                    result['probabilities'] = probs_dict

                results.append(result)

        return results

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'num_classes': self.model_config['num_classes'],
            'hidden_size': self.model_config['hidden_size'],
            'classifier_hidden_dim': self.model_config['classifier_hidden_dim'],
            'device': str(self.device),
            'classes': list(self.label_to_idx.keys()),
            'performance': self.performance,
            'model_path': str(self.model_path)
        }

    def print_info(self):
        """打印模型信息"""
        info = self.get_model_info()

        print("="*60)
        print("BERT分类器模型信息")
        print("="*60)
        print(f"类别数量: {info['num_classes']}")
        print(f"BERT向量维度: {info['hidden_size']}")
        print(f"分类器隐藏层: {info['classifier_hidden_dim']}")
        print(f"设备: {info['device']}")

        if info['performance']:
            print(f"\n训练性能:")
            print(f"  最佳验证准确率: {info['performance'].get('best_val_acc', 'N/A'):.2f}%")
            print(f"  最佳 Epoch: {info['performance'].get('best_epoch', 'N/A')}")

        print(f"\n支持的类别 ({info['num_classes']} 个):")
        for i, cls in enumerate(sorted(info['classes']), 1):
            print(f"  {i}. {cls}")

        print("="*60)


def load_bert_classifier(model_path: str, device: str = 'cuda') -> BertSpectrumClassifierInference:
    """
    便捷函数：加载训练好的BERT分类器

    Args:
        model_path: 模型文件路径
        device: 设备 ('cuda' 或 'cpu')

    Returns:
        BertSpectrumClassifierInference 实例
    """
    return BertSpectrumClassifierInference(model_path, device)


def visualize_prediction(prediction: Dict, top_k: int = 5):
    """
    可视化预测结果

    Args:
        prediction: 预测结果字典
        top_k: 显示前k个概率最高的类别
    """
    print("\n" + "="*60)
    print("预测结果")
    print("="*60)
    print(f"预测类别: {prediction['predicted_class']}")
    print(f"置信度: {prediction['confidence']:.4f} ({prediction['confidence']*100:.2f}%)")

    if 'probabilities' in prediction:
        print(f"\n前 {top_k} 个最可能的类别:")
        sorted_probs = sorted(
            prediction['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        for i, (cls, prob) in enumerate(sorted_probs, 1):
            bar = "█" * int(prob * 50)
            print(f"  {i}. {cls:30s} {prob:.4f} ({prob*100:5.2f}%) {bar}")

    print("="*60)
