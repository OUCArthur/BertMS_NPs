#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试注意力提取是否正常工作
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(os.getcwd())
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model import BertConfig, BertForMaskedLM_pre
from bertms_ws import Spec2BertVec
from matchms import Spectrum
from utils.attention_visualization import BertMSAttentionExtractor

print("="*60)
print("测试注意力提取")
print("="*60)

# 1. 加载模型
print("\n1. 加载模型...")
pretrained_model_dir = "bert_base_ms"
model_path = os.path.join(pretrained_model_dir, "cache_20240325_215535", "model_ms_225.bin")
config_path = os.path.join(pretrained_model_dir, "config.json")

config = BertConfig.from_json_file(config_path)
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForMaskedLM_pre.from_pretrained(config, model_path=model_path)
model = model.to(config.device)
model.eval()
print(f"✓ 模型加载成功")

# 2. 初始化编码器
spec2vec_encoder = Spec2BertVec(
    model=model,
    intensity_weighting_power=0.5,
    allowed_missing_percentage=5.0
)
print(f"✓ Spec2BertVec 初始化成功")

# 3. 创建测试质谱
print("\n2. 创建测试质谱...")
mz_array = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
intensity_array = np.array([10.0, 50.0, 100.0, 30.0, 20.0])

test_spectrum = Spectrum(
    mz=mz_array,
    intensities=intensity_array,
    metadata={'compound_name': 'Test Compound'}
)
print(f"✓ 测试质谱创建成功: {len(test_spectrum.peaks.mz)} 个峰")

# 4. 初始化注意力提取器
print("\n3. 初始化注意力提取器...")
attention_extractor = BertMSAttentionExtractor(model, config)

# 5. 提取注意力权重
print("\n4. 提取注意力权重...")
print("-"*60)
attention_matrix, peaks = attention_extractor.extract_attention_weights(
    spectrum=test_spectrum,
    spec2vec_encoder=spec2vec_encoder,
    layer_index=-1,
    head_index=None
)
print("-"*60)

# 6. 检查结果
print("\n5. 检查结果...")
print(f"✓ 注意力矩阵形状: {attention_matrix.shape}")
print(f"✓ 峰数量: {len(peaks)}")

# 检查注意力权重的统计信息
attn_numpy = attention_matrix.numpy()
print(f"\n注意力权重统计:")
print(f"  最小值: {attn_numpy.min():.6f}")
print(f"  最大值: {attn_numpy.max():.6f}")
print(f"  平均值: {attn_numpy.mean():.6f}")
print(f"  标准差: {attn_numpy.std():.6f}")

# 检查是否所有值都接近 1.0
if np.allclose(attn_numpy, 1.0, rtol=0.01):
    print("\n❌ 警告: 注意力权重几乎全是 1.0，提取可能失败！")
elif attn_numpy.std() < 0.01:
    print("\n❌ 警告: 注意力权重几乎没有变化，提取可能失败！")
else:
    print("\n✅ 注意力权重看起来正常，有明显的变化")

# 显示一些示例值
print(f"\n注意力矩阵示例（前5x5）:")
print(attn_numpy[:5, :5])

print("\n" + "="*60)
print("测试完成！")
print("="*60)
