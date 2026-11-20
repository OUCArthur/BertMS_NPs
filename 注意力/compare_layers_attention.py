#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比不同层的注意力分布，找出哪一层有明显的注意力模式
"""

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path

# 添加项目路径
project_root = Path(os.getcwd())
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model import BertConfig, BertForMaskedLM_pre
from bertms_ws import Spec2BertVec
from matchms import Spectrum
from utils.attention_visualization import BertMSAttentionExtractor

print("="*80)
print("对比不同BERT层的注意力分布")
print("="*80)

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
print(f"  总层数: {config.num_hidden_layers}")

# 2. 初始化编码器
spec2vec_encoder = Spec2BertVec(
    model=model,
    intensity_weighting_power=0.5,
    allowed_missing_percentage=5.0
)
print(f"✓ Spec2BertVec 初始化成功")

# 3. 加载测试质谱
print("\n2. 加载测试质谱...")
data_file = "data/ms/gnps_reclassified_1000_merged.json"

with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 找 Peptides 类别的第一个样本
peptide_samples = [item for item in data if item.get('detailed_class') == 'Peptides']
if not peptide_samples:
    print("⚠ 未找到 Peptides 样本，使用第一个样本")
    sample = data[0]
else:
    sample = peptide_samples[0]

peaks_data = sample.get('peaks_json', sample.get('peaks', []))
mz_array = np.array([float(p[0]) for p in peaks_data])
intensity_array = np.array([float(p[1]) for p in peaks_data])

test_spectrum = Spectrum(
    mz=mz_array,
    intensities=intensity_array,
    metadata={
        'compound_name': sample.get('compound_name', 'Unknown'),
        'compound_class': sample.get('detailed_class', 'Unknown')
    }
)
print(f"✓ 测试质谱: {test_spectrum.get('compound_name')}")
print(f"  类别: {test_spectrum.get('compound_class')}")
print(f"  峰数量: {len(test_spectrum.peaks.mz)}")

# 4. 初始化注意力提取器
print("\n3. 对比所有层的注意力分布...")
print("-"*80)

attention_extractor = BertMSAttentionExtractor(model, config)

layer_stats = []

# 对每一层提取注意力
for layer_idx in range(config.num_hidden_layers):
    print(f"\n{'='*80}")
    print(f"Layer {layer_idx}")
    print(f"{'='*80}")

    try:
        # 关闭调试输出（临时修改）
        import builtins
        original_print = builtins.print
        def quiet_print(*args, **kwargs):
            msg = ' '.join(str(arg) for arg in args)
            if '[调试]' not in msg and '✓' not in msg and '⚠' not in msg:
                original_print(*args, **kwargs)
        builtins.print = quiet_print

        attention_matrix, peaks = attention_extractor.extract_attention_weights(
            spectrum=test_spectrum,
            spec2vec_encoder=spec2vec_encoder,
            layer_index=layer_idx,
            head_index=None
        )

        # 恢复打印
        builtins.print = original_print

        # 统计注意力分布
        attn_numpy = attention_matrix.numpy()

        stats = {
            'layer': layer_idx,
            'min': float(attn_numpy.min()),
            'max': float(attn_numpy.max()),
            'mean': float(attn_numpy.mean()),
            'std': float(attn_numpy.std()),
            'range': float(attn_numpy.max() - attn_numpy.min()),
        }

        # 计算注意力熵（衡量均匀程度）
        # 熵越高，分布越均匀；熵越低，分布越集中
        flat_attn = attn_numpy.flatten()
        flat_attn = flat_attn / flat_attn.sum()  # 归一化为概率分布
        entropy = -np.sum(flat_attn * np.log(flat_attn + 1e-10))
        stats['entropy'] = float(entropy)

        layer_stats.append(stats)

        print(f"Layer {layer_idx} 统计:")
        print(f"  数值范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print(f"  均值: {stats['mean']:.6f}")
        print(f"  标准差: {stats['std']:.6f}")
        print(f"  熵: {stats['entropy']:.4f} (越高越均匀)")

        # 判断是否有明显的注意力模式
        if stats['std'] > 0.001:
            print(f"  ✅ 有明显的注意力模式！")
        else:
            print(f"  ❌ 注意力分布几乎均匀")

    except Exception as e:
        print(f"  ❌ 提取失败: {e}")
        builtins.print = original_print

print(f"\n{'='*80}")
print("汇总对比")
print(f"{'='*80}")

print(f"\n{'Layer':<8} {'Min':<12} {'Max':<12} {'Std':<12} {'Entropy':<12} {'状态'}")
print("-"*80)

for stats in layer_stats:
    status = "✅ 有模式" if stats['std'] > 0.001 else "❌ 均匀"
    print(f"{stats['layer']:<8} {stats['min']:<12.6f} {stats['max']:<12.6f} "
          f"{stats['std']:<12.6f} {stats['entropy']:<12.4f} {status}")

# 找出最有区分度的层
best_layer = max(layer_stats, key=lambda x: x['std'])
print(f"\n💡 推荐使用 Layer {best_layer['layer']}，标准差最大: {best_layer['std']:.6f}")

print(f"\n{'='*80}")
print("分析完成！")
print(f"{'='*80}")
