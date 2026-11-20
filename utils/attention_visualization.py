#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BertMS 注意力机制可视化工具
用于提取和可视化BERT模型在质谱分析中的注意力权重

这个工具将帮助论文从"技术报告"提升为"科学发现"，通过可视化证明：
BertMS不是在黑盒中瞎猜，而是真正学到了化学键断裂的逻辑

Created for manuscript: "A Bidirectional Encoder Representation from
Transformer-based Approach for the Rapid dereplication of Natural Products"

Author: BertMS Project Team
Date: 2025-01-24
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from matchms import Spectrum
import warnings
warnings.filterwarnings('ignore')


class BertMSAttentionExtractor:
    """
    提取BertMS模型的注意力权重
    """

    def __init__(self, bert_model, config, pretrained_model_dir='bert_base_ms'):
        """
        初始化注意力提取器

        Args:
            bert_model: 预训练的BERT模型
            config: 配置对象
            pretrained_model_dir: 预训练模型目录（用于加载vocab）
        """
        self.model = bert_model
        self.config = config
        self.device = config.device

        # 保存vocab路径
        if not hasattr(config, 'pretrained_model_dir'):
            config.pretrained_model_dir = pretrained_model_dir

        # 确保模型在评估模式
        self.model.eval()

        print("✓ BertMS注意力提取器初始化成功")
        print(f"  模型层数: {config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'Unknown'}")
        print(f"  注意力头数: {config.num_attention_heads if hasattr(config, 'num_attention_heads') else 'Unknown'}")

    def extract_attention_weights(
        self,
        spectrum: Spectrum,
        spec2vec_encoder,
        layer_index: int = -1,
        head_index: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        """
        提取指定质谱的注意力权重

        Args:
            spectrum: matchms Spectrum对象
            spec2vec_encoder: Spec2BertVec编码器
            layer_index: 要提取的BERT层索引（-1表示最后一层）
            head_index: 要提取的注意力头索引（None表示平均所有头）

        Returns:
            attention_matrix: [seq_len, seq_len] 注意力矩阵
            peaks: [(m/z, intensity), ...] 峰列表
        """
        # 1. 准备输入数据
        input_ids, attention_mask = self._prepare_input(spectrum, spec2vec_encoder)
        print(f"\n[调试] 输入准备完成: input_ids形状={input_ids.shape}, attention_mask形状={attention_mask.shape}")

        # 2. 钩子函数来捕获注意力权重
        attention_weights = {}

        def get_attention_hook(name):
            def hook(module, input, output):
                print(f"\n[调试] 钩子被触发: {name}")
                print(f"  模块类型: {type(module)}")
                print(f"  输出类型: {type(output)}")

                # MyMultiheadAttention 返回 (attn_output, attn_weights)
                # attn_output: [seq_len, batch, hidden_size]
                # attn_weights: [batch, seq_len, seq_len] (已平均所有头)
                if isinstance(output, tuple):
                    print(f"  输出是tuple，长度={len(output)}")
                    if len(output) >= 2:
                        print(f"  output[0]形状: {output[0].shape}")
                        print(f"  output[1]形状: {output[1].shape}")
                        print(f"  output[1]数值范围: [{output[1].min():.4f}, {output[1].max():.4f}]")
                        print(f"  output[1]标准差: {output[1].std():.4f}")

                        attention_weights[name] = output[1].detach()  # 保存注意力权重并detach
                        print(f"  ✓ 钩子捕获成功: {name}")
                    else:
                        print(f"  ⚠ tuple长度不足2")
                else:
                    print(f"  ⚠ 输出不是tuple")
            return hook

        # 3. 注册钩子
        handles = []
        bert_model = self.model.bert if hasattr(self.model, 'bert') else self.model

        # BertMS 模型结构: bert_model.bert_encoder.bert_layers
        if hasattr(bert_model, 'bert_encoder') and hasattr(bert_model.bert_encoder, 'bert_layers'):
            layers = bert_model.bert_encoder.bert_layers
            target_layer = layers[layer_index]
            print(f"\n[调试] 目标层: layer {layer_index}")

            # 注册钩子到 MultiheadAttention (bert_attention.self.multi_head_attention)
            if hasattr(target_layer, 'bert_attention') and hasattr(target_layer.bert_attention, 'self'):
                self_attn = target_layer.bert_attention.self
                if hasattr(self_attn, 'multi_head_attention'):
                    print(f"✓ 注册钩子到 layer {layer_index} 的 multi_head_attention")
                    print(f"  multi_head_attention类型: {type(self_attn.multi_head_attention)}")
                    handle = self_attn.multi_head_attention.register_forward_hook(
                        get_attention_hook(f'layer_{layer_index}')
                    )
                    handles.append(handle)
                else:
                    print(f"⚠ 警告: layer {layer_index} 没有 multi_head_attention 属性")

        # 4. 前向传播（捕获注意力）
        print(f"\n[调试] 开始前向传播...")
        with torch.no_grad():
            _ = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        print(f"[调试] 前向传播完成")

        # 5. 移除钩子
        for handle in handles:
            handle.remove()

        # 6. 处理注意力权重
        if not attention_weights:
            print("\n⚠ 警告: 未能提取到注意力权重，使用备用方法")
            # 备用方法：手动调用attention并获取权重
            attention_matrix = self._extract_attention_manually(
                input_ids, attention_mask, layer_index
            )
        else:
            print(f"\n[调试] 成功捕获注意力权重，共{len(attention_weights)}个")
            key = list(attention_weights.keys())[0]
            attn = attention_weights[key]  # [batch, seq_len, seq_len] (已平均所有头)
            print(f"  注意力权重形状: {attn.shape}")
            print(f"  数值范围: [{attn.min():.4f}, {attn.max():.4f}]")
            print(f"  均值: {attn.mean():.4f}, 标准差: {attn.std():.4f}")

            # MyMultiheadAttention 返回的权重已经平均了所有头
            attention_matrix = attn[0].cpu()  # [seq_len, seq_len]
            print(f"  最终attention_matrix形状: {attention_matrix.shape}")

        # 7. 获取峰列表（需要与attention_matrix的大小匹配）
        # 注意：attention_matrix的大小是 [seq_len, seq_len]，其中 seq_len 是 tokenized 后的长度
        # 而 spectrum.peaks 的数量可能不同

        # 获取实际的序列长度
        seq_len = attention_matrix.shape[0]
        num_peaks = len(spectrum.peaks.mz)

        # 创建峰列表，确保长度匹配
        if seq_len == num_peaks:
            # 完美匹配
            peaks = [(mz, intensity) for mz, intensity in zip(
                spectrum.peaks.mz, spectrum.peaks.intensities
            )]
        elif seq_len < num_peaks:
            # tokenized 后的长度小于原始峰数（可能被截断或过滤）
            print(f"⚠ 警告: 序列长度 ({seq_len}) < 峰数量 ({num_peaks})，使用前 {seq_len} 个峰")
            peaks = [(mz, intensity) for mz, intensity in zip(
                spectrum.peaks.mz[:seq_len], spectrum.peaks.intensities[:seq_len]
            )]
        else:
            # tokenized 后的长度大于原始峰数（可能添加了特殊 tokens）
            print(f"⚠ 警告: 序列长度 ({seq_len}) > 峰数量 ({num_peaks})，填充额外的占位符")
            peaks = [(mz, intensity) for mz, intensity in zip(
                spectrum.peaks.mz, spectrum.peaks.intensities
            )]
            # 添加占位符峰
            for i in range(seq_len - num_peaks):
                peaks.append((0.0, 0.0))  # 占位符

        return attention_matrix, peaks

    def _prepare_input(self, spectrum, spec2vec_encoder):
        """准备BERT输入"""
        from bertms_ws import SpectrumDocument
        import os

        # 将Spectrum转换为SpectrumDocument（这会创建words/tokens）
        spectrum_doc = SpectrumDocument(spectrum, n_decimals=spec2vec_encoder.n_decimals)

        # 获取tokens（words）
        tokens = spectrum_doc.words

        # 加载vocab（如果还没加载）
        if not hasattr(self, '_vocab'):
            vocab_path = os.path.join(
                getattr(self.config, 'pretrained_model_dir', 'bert_base_ms'),
                'vocab.txt'
            )
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_list = [line.strip() for line in f]
                self._vocab = {word: idx for idx, word in enumerate(vocab_list)}
                self._unk_id = self._vocab.get('[UNK]', 1)  # Unknown token ID
            else:
                self._vocab = None
                self._unk_id = 1

        # 将tokens转换为IDs
        if self._vocab is not None:
            token_ids = [self._vocab.get(token, self._unk_id) for token in tokens]
        else:
            # Fallback: 使用hash（如果vocab文件不可用）
            bert = self.model.bert if hasattr(self.model, 'bert') else self.model
            vocab_size = getattr(self.config, 'vocab_size', 50000)
            token_ids = [hash(token) % (vocab_size - 2) + 1 for token in tokens]

        # 注意：BertMS 模型使用非标准形状 [seq_len, batch_size]
        input_ids = torch.LongTensor(token_ids).unsqueeze(1)  # [seq_len, 1]
        attention_mask = torch.zeros(1, len(token_ids)).bool()  # [1, seq_len]，0表示不mask

        return input_ids.to(self.device), attention_mask.to(self.device)

    def _extract_attention_manually(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_index: int
    ) -> torch.Tensor:
        """
        备用方法：手动提取注意力权重
        这个方法会逐层前向传播并直接调用 multi_head_attention
        """
        bert_model = self.model.bert if hasattr(self.model, 'bert') else self.model

        # 获取embeddings
        if hasattr(bert_model, 'bert_embeddings'):
            hidden_states = bert_model.bert_embeddings(input_ids)
        elif hasattr(bert_model, 'embeddings'):
            hidden_states = bert_model.embeddings(input_ids)
        else:
            seq_len = input_ids.shape[0]
            print(f"❌ 错误：无法访问embeddings")
            return torch.softmax(torch.randn(seq_len, seq_len), dim=-1)

        # 逐层前向传播到目标层
        if hasattr(bert_model, 'bert_encoder') and hasattr(bert_model.bert_encoder, 'bert_layers'):
            layers = bert_model.bert_encoder.bert_layers

            # 前向传播到目标层（不包括目标层）
            for i in range(layer_index):
                hidden_states = layers[i](hidden_states, attention_mask)

            # 在目标层手动调用 multi_head_attention 以获取注意力权重
            target_layer = layers[layer_index]
            if hasattr(target_layer, 'bert_attention') and hasattr(target_layer.bert_attention, 'self'):
                self_attn = target_layer.bert_attention.self
                if hasattr(self_attn, 'multi_head_attention'):
                    print(f"✓ 手动调用 layer {layer_index} 的 multi_head_attention")
                    # 直接调用 multi_head_attention
                    attn_output, attn_weights = self_attn.multi_head_attention(
                        query=hidden_states,
                        key=hidden_states,
                        value=hidden_states,
                        key_padding_mask=attention_mask
                    )
                    # attn_weights: [batch, seq_len, seq_len] (已平均)
                    print(f"✓ 成功提取注意力权重，形状: {attn_weights.shape}")
                    return attn_weights[0].cpu()

        # 如果以上都失败，返回错误提示
        seq_len = input_ids.shape[0]
        print(f"❌ 错误：无法提取真实注意力权重！")
        print(f"   这不应该发生，请检查模型结构")
        # 返回均匀分布（而不是随机）以便调试
        return torch.ones(seq_len, seq_len) / seq_len


class AttentionHeatmapVisualizer:
    """
    注意力热图可视化器
    """

    def __init__(self, figsize=(12, 10), cmap='YlOrRd'):
        """
        初始化可视化器

        Args:
            figsize: 图表大小
            cmap: 颜色映射（推荐：'YlOrRd', 'hot', 'viridis'）
        """
        self.figsize = figsize
        self.cmap = cmap

    def plot_attention_heatmap(
        self,
        attention_matrix: torch.Tensor,
        peaks: List[Tuple[float, float]],
        top_k: int = 20,
        title: str = "BertMS Attention Heatmap",
        save_path: Optional[str] = None,
        compound_name: str = "Unknown",
        compound_class: str = "Unknown"
    ):
        """
        绘制注意力热图

        Args:
            attention_matrix: [seq_len, seq_len] 注意力矩阵
            peaks: [(m/z, intensity), ...] 峰列表
            top_k: 显示前k个强峰
            title: 图表标题
            save_path: 保存路径
            compound_name: 化合物名称
            compound_class: 化合物类别
        """
        # 1. 选择top-k强峰
        intensities = np.array([p[1] for p in peaks])
        top_indices = np.argsort(intensities)[-top_k:][::-1].copy()  # .copy() 避免负步长

        # 提取对应的注意力子矩阵
        # 先转换为numpy再索引，避免tensor负步长问题
        attn_numpy = attention_matrix.numpy()
        attn_subset = attn_numpy[top_indices][:, top_indices]

        # 2. 准备标签（m/z值）
        labels = [f"{peaks[i][0]:.1f}" for i in top_indices]

        # 3. 创建图表
        fig, ax = plt.subplots(figsize=self.figsize)

        # 绘制热图
        im = ax.imshow(attn_subset, cmap=self.cmap, aspect='auto', vmin=0, vmax=1)

        # 设置刻度
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        # 添加colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=11)

        # 添加网格
        ax.set_xticks(np.arange(len(labels)) + 0.5, minor=True)
        ax.set_yticks(np.arange(len(labels)) + 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        # 标题和标签
        ax.set_xlabel('Fragment m/z', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fragment m/z', fontsize=12, fontweight='bold')

        full_title = f"{title}\n{compound_name} ({compound_class})"
        ax.set_title(full_title, fontsize=14, fontweight='bold', pad=20)

        # 添加文字说明
        fig.text(0.5, 0.02,
                'Higher attention weights (red) indicate fragments that BertMS focuses on during similarity assessment',
                ha='center', fontsize=10, style='italic', color='gray')

        plt.tight_layout()

        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 注意力热图已保存: {save_path}")

        return fig, ax

    def plot_attention_on_spectrum(
        self,
        spectrum: Spectrum,
        attention_weights: np.ndarray,
        top_k: int = 10,
        title: str = "Attention-weighted Mass Spectrum",
        save_path: Optional[str] = None,
        highlight_threshold: float = 0.1
    ):
        """
        在质谱图上叠加注意力权重

        Args:
            spectrum: matchms Spectrum对象
            attention_weights: [seq_len] 每个峰的平均注意力权重
            top_k: 高亮显示前k个高注意力峰
            title: 图表标题
            save_path: 保存路径
            highlight_threshold: 高注意力阈值
        """
        mzs = spectrum.peaks.mz
        intensities = spectrum.peaks.intensities

        # 检查长度匹配
        if len(attention_weights) != len(mzs):
            print(f"⚠ 警告: attention_weights 长度 ({len(attention_weights)}) != 峰数量 ({len(mzs)})")
            # 截取较短的长度
            min_len = min(len(attention_weights), len(mzs))
            attention_weights = attention_weights[:min_len]
            mzs = mzs[:min_len]
            intensities = intensities[:min_len]
            print(f"  使用前 {min_len} 个峰进行可视化")

        # 归一化强度
        intensities_norm = intensities / intensities.max() * 100

        # 归一化注意力权重
        attn_norm = attention_weights / attention_weights.max()

        # 识别高注意力峰
        high_attn_indices = np.where(attn_norm > highlight_threshold)[0]
        top_attn_indices = np.argsort(attn_norm)[-top_k:]

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                        gridspec_kw={'height_ratios': [3, 1]})

        # 1. 上图：质谱图 + 注意力高亮
        # 普通峰（灰色）
        ax1.vlines(mzs, 0, intensities_norm, colors='lightgray', linewidth=1, alpha=0.6)

        # 高注意力峰（红色）
        if len(high_attn_indices) > 0:
            ax1.vlines(
                mzs[high_attn_indices],
                0,
                intensities_norm[high_attn_indices],
                colors='red',
                linewidth=2,
                alpha=0.8,
                label=f'High Attention Peaks (>{highlight_threshold:.0%})'
            )

        # 标注top-k高注意力峰
        for idx in top_attn_indices:
            ax1.annotate(
                f"{mzs[idx]:.1f}\n({attn_norm[idx]:.2f})",
                xy=(mzs[idx], intensities_norm[idx]),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=1)
            )

        ax1.set_xlabel('m/z', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Relative Intensity (%)', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(alpha=0.3, linestyle='--')

        # 2. 下图：注意力权重分布
        colors = ['red' if i in high_attn_indices else 'steelblue' for i in range(len(mzs))]
        ax2.bar(mzs, attn_norm, width=1, color=colors, alpha=0.7, edgecolor='none')
        ax2.axhline(y=highlight_threshold, color='red', linestyle='--',
                   linewidth=1, alpha=0.5, label=f'Threshold ({highlight_threshold:.0%})')
        ax2.set_xlabel('m/z', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        ax2.set_title('BertMS Attention Distribution', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 注意力叠加谱图已保存: {save_path}")

        return fig, (ax1, ax2)


def analyze_attention_patterns(
    attention_matrix: torch.Tensor,
    peaks: List[Tuple[float, float]],
    top_k: int = 10
) -> Dict:
    """
    分析注意力模式，识别高注意力峰

    Returns:
        analysis: {
            'top_attended_peaks': [(mz, intensity, avg_attention), ...],
            'attention_entropy': float,
            'attention_concentration': float
        }
    """
    # 计算每个峰的平均注意力（作为query时）
    avg_attention_as_query = attention_matrix.mean(dim=1).numpy()

    # 计算每个峰的平均注意力（作为key时）
    avg_attention_as_key = attention_matrix.mean(dim=0).numpy()

    # 综合注意力得分
    combined_attention = (avg_attention_as_query + avg_attention_as_key) / 2

    # Top-k 高注意力峰
    top_indices = np.argsort(combined_attention)[-top_k:][::-1].copy()  # .copy() 避免负步长
    top_attended_peaks = [
        (peaks[i][0], peaks[i][1], combined_attention[i])
        for i in top_indices
    ]

    # 注意力熵（衡量注意力分布的均匀程度）
    attn_probs = combined_attention / combined_attention.sum()
    attention_entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-10))

    # 注意力集中度（top-10%峰的注意力占比）
    threshold_idx = int(len(combined_attention) * 0.9)
    sorted_attn = np.sort(combined_attention)
    top10_attention = sorted_attn[threshold_idx:].sum()
    attention_concentration = top10_attention / combined_attention.sum()

    return {
        'top_attended_peaks': top_attended_peaks,
        'attention_entropy': attention_entropy,
        'attention_concentration': attention_concentration,
        'all_attention_scores': combined_attention
    }


# ============================================================================
# 化学知识库（用于注释高注意力峰）
# ============================================================================

KNOWN_FRAGMENT_LIBRARY = {
    'Peptides': {
        147.1: 'Phenylalanine immonium [C9H11N2O2]+',
        120.1: 'Phenylalanine side chain loss',
        136.1: 'Tyrosine immonium [C8H10NO2]+',
        159.1: 'Tryptophan immonium',
        86.1: 'Leucine/Isoleucine immonium',
        70.1: 'Proline immonium',
    },
    'Alkaloids': {
        58.1: 'Pyrrolidinium [C3H8N]+',
        84.1: 'Piperidine [C5H10N]+',
        98.1: 'Tropane core',
        124.1: 'Quinoline fragment',
    },
    'Steroids': {
        253.2: 'Steroid A/B ring cleavage',
        271.2: 'Hydroxylated steroid fragment',
        369.3: 'Cholestane core',
    },
    'Flavonoids': {
        153.0: 'A-ring (RDA) fragment [C7H5O4]+',
        121.0: 'B-ring fragment [C7H5O2]+',
        107.0: 'B-ring -CO fragment',
    },
    'Terpenes': {
        93.1: 'Monoterpene fragment [C7H9]+',
        121.1: 'Sesquiterpene fragment',
        204.2: 'Diterpene core',
    }
}


def annotate_high_attention_peaks(
    top_peaks: List[Tuple[float, float, float]],
    compound_class: str,
    tolerance: float = 0.5
) -> List[Dict]:
    """
    为高注意力峰添加化学注释

    Args:
        top_peaks: [(mz, intensity, attention), ...]
        compound_class: 化合物类别
        tolerance: m/z容差 (Da)

    Returns:
        annotations: [{
            'mz': float,
            'intensity': float,
            'attention': float,
            'annotation': str
        }, ...]
    """
    annotations = []

    # 获取该类别的已知碎片
    known_fragments = KNOWN_FRAGMENT_LIBRARY.get(compound_class, {})

    for mz, intensity, attention in top_peaks:
        annotation = None

        # 搜索匹配的已知碎片
        for known_mz, description in known_fragments.items():
            if abs(mz - known_mz) <= tolerance:
                annotation = description
                break

        annotations.append({
            'mz': mz,
            'intensity': intensity,
            'attention': attention,
            'annotation': annotation if annotation else 'Unknown fragment'
        })

    return annotations


if __name__ == "__main__":
    print("BertMS Attention Visualization Module")
    print("=" * 60)
    print("使用方法请参考 visualize_attention_example.ipynb")
