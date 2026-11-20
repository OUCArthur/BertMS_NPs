"""
深度学习质谱分类器推理模块

用于加载训练好的深度学习模型并进行推理
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.spectrum_classifier import create_model


class DeepLearningClassifierPredictor:
    """深度学习分类器预测器"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化预测器

        Args:
            model_path: 训练好的模型路径
            device: 设备 ('cuda' 或 'cpu')
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model, self.config, self.label_mapping = self._load_model()

        print(f"✓ 深度学习分类器已加载")
        print(f"  模型类型: {self.config['model_type']}")
        print(f"  设备: {self.device}")
        print(f"  类别数: {self.config['num_classes']}")
        print(f"  类别: {list(self.label_mapping['idx_to_label'].values())}")
        print(f"  最佳验证准确率: {self.config.get('best_val_acc', 'N/A')}")

    def _load_model(self):
        """加载模型"""
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 提取配置
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = checkpoint.get('model_config', {})

        # 创建模型
        model_config = checkpoint['model_config']
        model = create_model(
            model_type=checkpoint['model_type'],
            num_classes=model_config['num_classes'],
            embedding_dim=model_config.get('embedding_dim', 128)
        )

        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        # 标签映射
        label_mapping = checkpoint['label_mapping']

        # 性能信息
        if 'performance' in checkpoint:
            config['best_val_acc'] = checkpoint['performance'].get('best_val_acc', 'N/A')

        return model, config, label_mapping

    def _process_spectrum(self, spectrum) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理质谱数据

        Args:
            spectrum: 质谱对象（matchms Spectrum）

        Returns:
            mz: (1, max_peaks, 1)
            intensity: (1, max_peaks, 1)
            mask: (1, max_peaks)
        """
        max_peaks = self.config['max_peaks']

        # 获取峰数据
        if not hasattr(spectrum, 'peaks') or len(spectrum.peaks.mz) == 0:
            # 空质谱
            mz = np.zeros((1, max_peaks, 1), dtype=np.float32)
            intensity = np.zeros((1, max_peaks, 1), dtype=np.float32)
            mask = np.zeros((1, max_peaks), dtype=np.float32)
            return (
                torch.from_numpy(mz).to(self.device),
                torch.from_numpy(intensity).to(self.device),
                torch.from_numpy(mask).to(self.device)
            )

        mz_values = spectrum.peaks.mz
        intensities = spectrum.peaks.intensities

        # 按强度排序，保留top-k峰
        if len(mz_values) > max_peaks:
            top_indices = np.argsort(intensities)[-max_peaks:]
            mz_values = mz_values[top_indices]
            intensities = intensities[top_indices]

        # 按m/z排序
        sort_indices = np.argsort(mz_values)
        mz_values = mz_values[sort_indices]
        intensities = intensities[sort_indices]

        num_peaks = len(mz_values)

        # 归一化m/z到[0, 1]（假设范围0-2000）
        mz_normalized = mz_values / 2000.0
        mz_normalized = np.clip(mz_normalized, 0, 1)

        # 归一化强度
        if intensities.max() > 0:
            intensity_normalized = intensities / intensities.max()
        else:
            intensity_normalized = intensities

        # Padding
        mz_padded = np.zeros((1, max_peaks, 1), dtype=np.float32)
        intensity_padded = np.zeros((1, max_peaks, 1), dtype=np.float32)
        mask = np.zeros((1, max_peaks), dtype=np.float32)

        mz_padded[0, :num_peaks, 0] = mz_normalized
        intensity_padded[0, :num_peaks, 0] = intensity_normalized
        mask[0, :num_peaks] = 1.0

        return (
            torch.from_numpy(mz_padded).to(self.device),
            torch.from_numpy(intensity_padded).to(self.device),
            torch.from_numpy(mask).to(self.device)
        )

    def predict(self, spectrum) -> Tuple[str, float]:
        """
        预测质谱的结构类别

        Args:
            spectrum: 质谱对象

        Returns:
            predicted_class: 预测的类别名称
            confidence: 置信度（概率）
        """
        try:
            # 处理质谱
            mz, intensity, mask = self._process_spectrum(spectrum)

            # 推理
            with torch.no_grad():
                logits = self.model(mz, intensity, mask)
                probabilities = torch.softmax(logits, dim=1)
                confidence, pred_idx = probabilities.max(1)

                pred_idx = pred_idx.item()
                confidence = confidence.item()

            # 获取类别名称
            predicted_class = self.label_mapping['idx_to_label'][str(pred_idx)]

            return predicted_class, confidence

        except Exception as e:
            print(f"深度学习预测失败: {e}")
            import traceback
            traceback.print_exc()
            return 'Unknown', 0.0

    def predict_batch(self, spectrums):
        """
        批量预测

        Args:
            spectrums: 质谱对象列表

        Returns:
            predictions: [(class, confidence), ...]
        """
        predictions = []
        for spectrum in spectrums:
            pred_class, confidence = self.predict(spectrum)
            predictions.append((pred_class, confidence))
        return predictions


if __name__ == '__main__':
    print("深度学习分类器推理模块")
    print("\n用法:")
    print("  predictor = DeepLearningClassifierPredictor('results/dl_models/spectrum_dl_classifier.pth')")
    print("  predicted_class, confidence = predictor.predict(spectrum)")
