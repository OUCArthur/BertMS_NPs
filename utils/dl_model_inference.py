#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度学习模型推理工具
用于在 Jupyter Notebook 中加载和使用训练好的质谱分类模型

Created by: BertMS Project
Date: 2025-01-24
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.spectrum_classifier import create_model


class SpectrumDLClassifier:
    """深度学习质谱分类器推理类"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化分类器

        Args:
            model_path: 模型文件路径 (.pth)
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)

        # 加载模型数据
        print(f"加载模型: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 提取配置信息
        self.model_type = checkpoint['model_type']
        self.model_config = checkpoint['model_config']
        self.label_mapping = checkpoint['label_mapping']
        self.performance = checkpoint.get('performance', {})
        self.training_config = checkpoint.get('config', {})

        # 创建模型
        self.model = create_model(
            model_type=self.model_type,
            num_classes=self.model_config['num_classes'],
            embedding_dim=self.model_config['embedding_dim'],
            dropout=0.0  # 推理时不需要 dropout
        )

        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 标签映射
        self.label_to_idx = self.label_mapping['label_to_idx']
        self.idx_to_label = {int(k): v for k, v in self.label_mapping['idx_to_label'].items()}

        print(f"✓ 模型加载成功")
        print(f"  模型类型: {self.model_type}")
        print(f"  类别数量: {self.model_config['num_classes']}")
        print(f"  设备: {self.device}")
        if self.performance:
            print(f"  训练性能: {self.performance.get('best_val_acc', 'N/A'):.2f}% (Epoch {self.performance.get('best_epoch', 'N/A')})")

    def preprocess_spectrum(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        max_peaks: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预处理单个质谱

        Args:
            mz: m/z 数组
            intensity: 强度数组
            max_peaks: 最大峰数（如果为None，使用训练时的设置）

        Returns:
            (mz_tensor, intensity_tensor, mask_tensor)
        """
        if max_peaks is None:
            max_peaks = self.training_config.get('max_peaks', 200)

        # 确保是numpy数组
        mz = np.array(mz)
        intensity = np.array(intensity)

        # 归一化强度
        if intensity.max() > 0:
            intensity = intensity / intensity.max()

        # 按强度排序，保留top-k峰
        if len(mz) > max_peaks:
            top_indices = np.argsort(intensity)[-max_peaks:]
            mz = mz[top_indices]
            intensity = intensity[top_indices]

        # 按m/z排序
        sort_indices = np.argsort(mz)
        mz = mz[sort_indices]
        intensity = intensity[sort_indices]

        # Padding
        actual_length = len(mz)
        if actual_length < max_peaks:
            mz = np.pad(mz, (0, max_peaks - actual_length), constant_values=0)
            intensity = np.pad(intensity, (0, max_peaks - actual_length), constant_values=0)

        # 创建mask (1=有效峰, 0=padding)
        mask = np.zeros(max_peaks, dtype=np.float32)
        mask[:actual_length] = 1.0

        # 转换为tensor，注意形状需要是 (1, max_peaks, 1)
        mz_tensor = torch.FloatTensor(mz).unsqueeze(0).unsqueeze(-1)  # (1, max_peaks, 1)
        intensity_tensor = torch.FloatTensor(intensity).unsqueeze(0).unsqueeze(-1)  # (1, max_peaks, 1)
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0)  # (1, max_peaks)

        # 调试：验证形状
        assert mz_tensor.shape[2] == 1, f"mz_tensor shape错误: {mz_tensor.shape}, 期望最后一维是1"
        assert intensity_tensor.shape[2] == 1, f"intensity_tensor shape错误: {intensity_tensor.shape}, 期望最后一维是1"

        return mz_tensor, intensity_tensor, mask_tensor

    def preprocess_batch(
        self,
        spectrums: List[Tuple[np.ndarray, np.ndarray]],
        max_peaks: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量预处理质谱

        Args:
            spectrums: List of (mz, intensity) tuples
            max_peaks: 最大峰数

        Returns:
            (batch_mz, batch_intensity, batch_mask)
        """
        mz_list = []
        intensity_list = []
        mask_list = []

        for mz, intensity in spectrums:
            mz_t, int_t, mask_t = self.preprocess_spectrum(mz, intensity, max_peaks)
            mz_list.append(mz_t)
            intensity_list.append(int_t)
            mask_list.append(mask_t)

        batch_mz = torch.cat(mz_list, dim=0)
        batch_intensity = torch.cat(intensity_list, dim=0)
        batch_mask = torch.cat(mask_list, dim=0)

        return batch_mz, batch_intensity, batch_mask

    @torch.no_grad()
    def predict(
        self,
        mz: np.ndarray,
        intensity: np.ndarray,
        return_probabilities: bool = False
    ) -> Dict:
        """
        预测单个质谱的化合物类别

        Args:
            mz: m/z 数组
            intensity: 强度数组
            return_probabilities: 是否返回所有类别的概率

        Returns:
            预测结果字典，包含:
                - predicted_class: 预测的类别名称
                - confidence: 预测置信度 (0-1)
                - class_idx: 类别索引
                - probabilities: 所有类别的概率 (如果 return_probabilities=True)
        """
        # 预处理
        mz_tensor, intensity_tensor, mask_tensor = self.preprocess_spectrum(mz, intensity)

        # 移到设备
        mz_tensor = mz_tensor.to(self.device)
        intensity_tensor = intensity_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)

        # 前向传播
        outputs = self.model(mz_tensor, intensity_tensor, mask_tensor)

        # 计算概率
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

        # 转换为numpy
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
    def predict_batch(
        self,
        spectrums: List[Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 32,
        return_probabilities: bool = False
    ) -> List[Dict]:
        """
        批量预测质谱

        Args:
            spectrums: List of (mz, intensity) tuples
            batch_size: 批大小
            return_probabilities: 是否返回概率

        Returns:
            预测结果列表
        """
        results = []

        for i in range(0, len(spectrums), batch_size):
            batch_spectrums = spectrums[i:i+batch_size]

            # 预处理
            mz_tensor, intensity_tensor, mask_tensor = self.preprocess_batch(batch_spectrums)

            # 移到设备
            mz_tensor = mz_tensor.to(self.device)
            intensity_tensor = intensity_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)

            # 调试：打印形状
            print(f"DEBUG - Batch shapes:")
            print(f"  mz_tensor: {mz_tensor.shape}")
            print(f"  intensity_tensor: {intensity_tensor.shape}")
            print(f"  mask_tensor: {mask_tensor.shape}")

            # 前向传播
            outputs = self.model(mz_tensor, intensity_tensor, mask_tensor)

            # 计算概率
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted_indices = probabilities.max(1)

            # 转换结果
            for j in range(len(batch_spectrums)):
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
            'model_type': self.model_type,
            'num_classes': self.model_config['num_classes'],
            'embedding_dim': self.model_config['embedding_dim'],
            'max_peaks': self.model_config['max_peaks'],
            'device': str(self.device),
            'classes': list(self.label_to_idx.keys()),
            'performance': self.performance,
            'model_path': str(self.model_path)
        }

    def print_info(self):
        """打印模型信息"""
        info = self.get_model_info()

        print("="*60)
        print("模型信息")
        print("="*60)
        print(f"模型类型: {info['model_type']}")
        print(f"类别数量: {info['num_classes']}")
        print(f"嵌入维度: {info['embedding_dim']}")
        print(f"最大峰数: {info['max_peaks']}")
        print(f"设备: {info['device']}")

        if info['performance']:
            print(f"\n训练性能:")
            print(f"  最佳验证准确率: {info['performance'].get('best_val_acc', 'N/A'):.2f}%")
            print(f"  最佳 Epoch: {info['performance'].get('best_epoch', 'N/A')}")

        print(f"\n支持的类别 ({info['num_classes']} 个):")
        for i, cls in enumerate(sorted(info['classes']), 1):
            print(f"  {i}. {cls}")

        print("="*60)


def load_classifier(model_path: str, device: str = 'cuda') -> SpectrumDLClassifier:
    """
    便捷函数：加载训练好的分类器

    Args:
        model_path: 模型文件路径
        device: 设备 ('cuda' 或 'cpu')

    Returns:
        SpectrumDLClassifier 实例
    """
    return SpectrumDLClassifier(model_path, device)


# ============= Jupyter Notebook 便捷函数 =============

def predict_from_matchms_spectrum(
    classifier: SpectrumDLClassifier,
    spectrum,
    return_probabilities: bool = False
) -> Dict:
    """
    从 matchms Spectrum 对象预测

    Args:
        classifier: SpectrumDLClassifier 实例
        spectrum: matchms Spectrum 对象
        return_probabilities: 是否返回概率

    Returns:
        预测结果
    """
    mz = spectrum.peaks.mz
    intensity = spectrum.peaks.intensities

    return classifier.predict(mz, intensity, return_probabilities)


def predict_batch_from_matchms(
    classifier: SpectrumDLClassifier,
    spectrums: List,
    batch_size: int = 32,
    return_probabilities: bool = False
) -> List[Dict]:
    """
    批量从 matchms Spectrum 对象预测

    Args:
        classifier: SpectrumDLClassifier 实例
        spectrums: matchms Spectrum 对象列表
        batch_size: 批大小
        return_probabilities: 是否返回概率

    Returns:
        预测结果列表
    """
    spectrum_data = [
        (s.peaks.mz, s.peaks.intensities)
        for s in spectrums
    ]

    return classifier.predict_batch(
        spectrum_data,
        batch_size=batch_size,
        return_probabilities=return_probabilities
    )


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
