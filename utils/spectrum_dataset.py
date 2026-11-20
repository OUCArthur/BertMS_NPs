#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
质谱数据集处理

用于深度学习模型的数据加载和预处理
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class SpectrumDataset(Dataset):
    """质谱数据集类"""

    def __init__(self, data_path: str, max_peaks: int = 200,
                 mz_range: Tuple[float, float] = (0, 2000),
                 normalize_intensity: bool = True):
        """
        初始化质谱数据集

        Args:
            data_path: GNPS数据JSON文件路径
            max_peaks: 最大峰数（超过则截断，不足则padding）
            mz_range: m/z值范围，用于归一化
            normalize_intensity: 是否归一化强度
        """
        self.max_peaks = max_peaks
        self.mz_range = mz_range
        self.normalize_intensity = normalize_intensity

        # 加载数据
        self.data = self._load_data(data_path)

        # 构建标签映射
        self.label_to_idx, self.idx_to_label = self._build_label_mapping()

        print(f"数据集加载完成:")
        print(f"  样本数: {len(self.data)}")
        print(f"  类别数: {len(self.label_to_idx)}")
        print(f"  类别分布: {self._get_class_distribution()}")

    def _load_data(self, data_path: str) -> List[Dict]:
        """加载GNPS数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 过滤有效数据（必须有peaks和label）
        # 支持 detailed_class（新分类）或 library_class（旧分类）
        valid_data = []
        for item in data:
            if 'peaks_json' in item and item['peaks_json']:
                # 优先使用 detailed_class，如果没有则使用 library_class
                if 'detailed_class' in item or 'library_class' in item:
                    valid_data.append(item)

        return valid_data

    def _build_label_mapping(self) -> Tuple[Dict, Dict]:
        """构建标签映射"""
        # 优先使用 detailed_class，如果没有则使用 library_class
        unique_labels = sorted(set(
            item.get('detailed_class', item.get('library_class')) for item in self.data
        ))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        return label_to_idx, idx_to_label

    def _get_class_distribution(self) -> Dict:
        """获取类别分布"""
        from collections import Counter
        # 优先使用 detailed_class，如果没有则使用 library_class
        labels = [item.get('detailed_class', item.get('library_class')) for item in self.data]
        return dict(Counter(labels))

    def _process_peaks(self, peaks_json: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        处理质谱峰数据

        Args:
            peaks_json: [[m/z, intensity], ...] 格式的峰列表

        Returns:
            mz_values: (max_peaks,) 归一化的m/z值
            intensities: (max_peaks,) 归一化的强度值
            mask: (max_peaks,) 有效峰位置为1，padding为0
        """
        # 转换为numpy数组
        peaks = np.array(peaks_json, dtype=np.float32)

        if len(peaks) == 0:
            # 空质谱
            mz_values = np.zeros(self.max_peaks, dtype=np.float32)
            intensities = np.zeros(self.max_peaks, dtype=np.float32)
            mask = np.zeros(self.max_peaks, dtype=np.float32)
            return mz_values, intensities, mask

        mz = peaks[:, 0]
        intensity = peaks[:, 1]

        # 按强度排序，保留top-k峰
        if len(peaks) > self.max_peaks:
            top_indices = np.argsort(intensity)[-self.max_peaks:]
            mz = mz[top_indices]
            intensity = intensity[top_indices]

        # 按m/z排序
        sort_indices = np.argsort(mz)
        mz = mz[sort_indices]
        intensity = intensity[sort_indices]

        num_peaks = len(mz)

        # 归一化m/z到[0, 1]
        mz_normalized = (mz - self.mz_range[0]) / (self.mz_range[1] - self.mz_range[0])
        mz_normalized = np.clip(mz_normalized, 0, 1)

        # 归一化强度
        if self.normalize_intensity and intensity.max() > 0:
            intensity_normalized = intensity / intensity.max()
        else:
            intensity_normalized = intensity

        # Padding
        mz_padded = np.zeros(self.max_peaks, dtype=np.float32)
        intensity_padded = np.zeros(self.max_peaks, dtype=np.float32)
        mask = np.zeros(self.max_peaks, dtype=np.float32)

        mz_padded[:num_peaks] = mz_normalized
        intensity_padded[:num_peaks] = intensity_normalized
        mask[:num_peaks] = 1.0

        return mz_padded, intensity_padded, mask

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            {
                'mz': (max_peaks, 1),
                'intensity': (max_peaks, 1),
                'mask': (max_peaks,),
                'label': scalar,
                'label_name': str
            }
        """
        item = self.data[idx]

        # 处理峰数据
        mz, intensity, mask = self._process_peaks(item['peaks_json'])

        # 获取标签 - 优先使用 detailed_class，如果没有则使用 library_class
        label_name = item.get('detailed_class', item.get('library_class'))
        label = self.label_to_idx[label_name]

        return {
            'mz': torch.FloatTensor(mz).unsqueeze(-1),  # (max_peaks, 1)
            'intensity': torch.FloatTensor(intensity).unsqueeze(-1),  # (max_peaks, 1)
            'mask': torch.FloatTensor(mask),  # (max_peaks,)
            'label': torch.LongTensor([label])[0],  # scalar
            'label_name': label_name,
            'compound_name': item.get('compound_name', 'Unknown')
        }


def create_dataloaders(data_path: str, batch_size: int = 32,
                      train_split: float = 0.8, max_peaks: int = 200,
                      num_workers: int = 0, random_seed: int = 42) -> Tuple[DataLoader, DataLoader, SpectrumDataset]:
    """
    创建训练和验证数据加载器

    Args:
        data_path: 数据文件路径
        batch_size: 批大小
        train_split: 训练集比例
        max_peaks: 最大峰数
        num_workers: 数据加载线程数
        random_seed: 随机种子

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        dataset: 完整数据集（用于获取标签映射）
    """
    # 创建完整数据集
    dataset = SpectrumDataset(data_path, max_peaks=max_peaks)

    # 分割训练集和验证集
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # 设置随机种子
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    split_idx = int(train_split * dataset_size)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # 创建采样器
    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\n数据加载器创建完成:")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本")

    return train_loader, val_loader, dataset


if __name__ == '__main__':
    # 测试
    print("测试质谱数据集...")

    data_path = "../data/ms/gnps_all_cleand_by_matchms_and_pubchem_lookups_1000.json"

    if Path(data_path).exists():
        # 创建数据集
        dataset = SpectrumDataset(data_path, max_peaks=100)

        # 测试单个样本
        sample = dataset[0]
        print(f"\n样本示例:")
        print(f"  m/z shape: {sample['mz'].shape}")
        print(f"  intensity shape: {sample['intensity'].shape}")
        print(f"  mask shape: {sample['mask'].shape}")
        print(f"  label: {sample['label']}")
        print(f"  label_name: {sample['label_name']}")
        print(f"  compound: {sample['compound_name'][:50]}")

        # 创建数据加载器
        train_loader, val_loader, _ = create_dataloaders(
            data_path, batch_size=4, train_split=0.8, max_peaks=100
        )

        # 测试批次
        batch = next(iter(train_loader))
        print(f"\n批次示例:")
        print(f"  m/z shape: {batch['mz'].shape}")
        print(f"  intensity shape: {batch['intensity'].shape}")
        print(f"  mask shape: {batch['mask'].shape}")
        print(f"  label shape: {batch['label'].shape}")
    else:
        print(f"数据文件不存在: {data_path}")
