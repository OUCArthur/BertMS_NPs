#!/usr/bin/env python3
"""
检查所有层的注意力模式，找到最有区分度的层
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无GUI模式
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
project_root = Path(os.getcwd())
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入模块
from model import BertConfig, BertForMaskedLM_pre
from bertms_ws import Spec2BertVec
from utils.attention_visualization import BertMSAttentionExtractor
import json

# ============================================================================
# 1. 加载模型
# ============================================================================
print("="*80)
print("加载BertMS模型")
print("="*80)

pretrained_model_dir = "bert_base_ms"
model_path = os.path.join(pretrained_model_dir, "cache_20240325_215535", "model_ms_225.bin")
config_path = os.path.join(pretrained_model_dir, "config.json")

config = BertConfig.from_json_file(config_path)
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"设备: {config.device}")
print(f"编码器层数: {config.num_hidden_layers}")

model = BertForMaskedLM_pre.from_pretrained(config, model_path=model_path)
model = model.to(config.device)
model.eval()

spec2vec_encoder = Spec2BertVec(
    model=model,
    intensity_weighting_power=0.5,
    allowed_missing_percentage=5.0
)

attention_extractor = BertMSAttentionExtractor(model, config)

print("✓ 模型加载完成")

# ============================================================================
# 2. 加载示例质谱
# ============================================================================
print("\n" + "="*80)
print("加载示例质谱")
print("="*80)

data_file = "data/ms/gnps_reclassified_1000.json"

with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 选择第一个样本
sample = data[0]
peaks_data = sample.get('peaks_json', sample.get('peaks', []))
mz_array = np.array([float(p[0]) for p in peaks_data])
intensity_array = np.array([float(p[1]) for p in peaks_data])

from matchms import Spectrum
sample_spectrum = Spectrum(
    mz=mz_array,
    intensities=intensity_array,
    metadata={
        'compound_name': sample.get('compound_name', 'Unknown'),
        'compound_class': sample.get('detailed_class', 'Unknown'),
        'precursor_mz': sample.get('precursor_mz', 0)
    }
)

print(f"✓ 加载质谱: {sample_spectrum.get('compound_name')}")
print(f"  类别: {sample_spectrum.get('compound_class')}")
print(f"  峰数量: {len(sample_spectrum.peaks.mz)}")

# ============================================================================
# 3. 检查所有层的注意力模式
# ============================================================================
print("\n" + "="*80)
print("多层注意力诊断 - 检查哪一层的注意力最有区分度")
print("="*80)

num_layers = config.num_hidden_layers
layer_stats = []

for layer_idx in range(num_layers):
    try:
        print(f"\n检查 Layer {layer_idx}...", end=' ', flush=True)

        attn_mat, _ = attention_extractor.extract_attention_weights(
            spectrum=sample_spectrum,
            spec2vec_encoder=spec2vec_encoder,
            layer_index=layer_idx,
            head_index=None
        )

        attn_np = attn_mat.numpy()
        std = attn_np.std()
        mean = attn_np.mean()

        # 计算熵
        row_entropies = []
        for i in range(attn_np.shape[0]):
            row = attn_np[i]
            row_prob = row / (row.sum() + 1e-10)
            entropy = -np.sum(row_prob * np.log(row_prob + 1e-10))
            row_entropies.append(entropy)
        avg_entropy = np.mean(row_entropies)
        max_entropy = np.log(attn_np.shape[0])
        norm_entropy = avg_entropy / max_entropy

        layer_stats.append({
            'layer': layer_idx,
            'std': std,
            'mean': mean,
            'avg_entropy': avg_entropy,
            'norm_entropy': norm_entropy
        })

        print(f"std={std:.6f}, norm_entropy={norm_entropy:.2%}")

    except Exception as e:
        print(f"失败: {e}")
        continue

# ============================================================================
# 4. 汇总结果
# ============================================================================
if layer_stats:
    print("\n" + "="*80)
    print("各层注意力统计汇总")
    print("="*80)

    df_layers = pd.DataFrame(layer_stats)
    print(df_layers.to_string(index=False))

    # 保存到CSV
    os.makedirs('results/manuscript_figures', exist_ok=True)
    df_layers.to_csv('results/manuscript_figures/layer_attention_stats.csv', index=False)
    print("\n✓ 统计数据已保存: results/manuscript_figures/layer_attention_stats.csv")

    # 找到最好的层
    best_layer = df_layers.loc[df_layers['std'].idxmax()]
    print(f"\n" + "="*80)
    print("最佳层推荐")
    print("="*80)
    print(f"✓ 最佳层: Layer {int(best_layer['layer'])}")
    print(f"  标准差: {best_layer['std']:.6f}")
    print(f"  归一化熵: {best_layer['norm_entropy']:.2%}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df_layers['layer'], df_layers['std'], marker='o', linewidth=2, markersize=8)
    axes[0].axhline(y=0.01, color='red', linestyle='--', label='阈值 (0.01)')
    axes[0].set_xlabel('Layer Index', fontsize=11)
    axes[0].set_ylabel('Standard Deviation', fontsize=11)
    axes[0].set_title('Attention Std across Layers', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df_layers['layer'], df_layers['norm_entropy'], marker='s', linewidth=2, markersize=8, color='green')
    axes[1].axhline(y=0.9, color='red', linestyle='--', label='阈值 (0.9)')
    axes[1].set_xlabel('Layer Index', fontsize=11)
    axes[1].set_ylabel('Normalized Entropy', fontsize=11)
    axes[1].set_title('Attention Entropy across Layers', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/manuscript_figures/attention_layer_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ 可视化图表已保存: results/manuscript_figures/attention_layer_comparison.png")

    # 判断是否找到有意义的层
    if best_layer['std'] > 0.01:
        print(f"\n✅ 找到有意义的注意力层!")
        print(f"👉 建议使用 Layer {int(best_layer['layer'])} 进行后续可视化")

        # 保存推荐层到文件
        with open('results/manuscript_figures/recommended_layer.txt', 'w') as f:
            f.write(f"{int(best_layer['layer'])}\n")
        print("\n✓ 推荐层已保存到: results/manuscript_figures/recommended_layer.txt")
    else:
        print("\n❌ 所有层的注意力都缺乏区分度")
        print("   可能需要:")
        print("   1. 检查模型加载是否正确")
        print("   2. 检查模型配置（use_torch_multi_head）")
        print("   3. 考虑使用其他可解释性方法（梯度、SHAP）")

    print("\n" + "="*80)
else:
    print("\n❌ 无法提取注意力权重")

print("\n✓ 诊断完成!")
