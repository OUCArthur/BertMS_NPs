#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度学习质谱分类模型

支持多种现代深度学习架构:
1. Spec2Class-CNN: 1D卷积神经网络
2. Spec2Class-Transformer: 自注意力机制
3. Spec2Class-Hybrid: CNN + Attention混合模型（推荐）
4. Spec2Class-ResNet: 残差网络

Created by: BertMS Project
Date: 2025-01-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PeakEmbedding(nn.Module):
    """质谱峰嵌入层

    将每个(m/z, intensity)峰转换为嵌入向量
    """

    def __init__(self, embedding_dim: int = 128):
        super(PeakEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        # m/z嵌入（使用位置编码思想）
        self.mz_projection = nn.Linear(1, embedding_dim // 2)

        # intensity嵌入
        self.intensity_projection = nn.Linear(1, embedding_dim // 2)

        # 组合投影
        self.combine = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, mz_values: torch.Tensor, intensities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mz_values: (batch_size, num_peaks, 1)
            intensities: (batch_size, num_peaks, 1)

        Returns:
            embeddings: (batch_size, num_peaks, embedding_dim)
        """
        # 投影m/z和强度
        mz_embed = self.mz_projection(mz_values)  # (B, N, D/2)
        intensity_embed = self.intensity_projection(intensities)  # (B, N, D/2)

        # 拼接
        combined = torch.cat([mz_embed, intensity_embed], dim=-1)  # (B, N, D)

        # 组合投影和归一化
        output = self.combine(combined)
        output = self.layer_norm(output)

        return output


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) - 0表示padding位置

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        residual = x

        # 线性投影
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 应用mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        context = torch.matmul(attn_weights, V)  # (B, num_heads, seq_len, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 输出投影
        output = self.out_linear(context)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)

        return output


class Conv1DBlock(nn.Module):
    """1D卷积块（含残差连接）"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout: float = 0.1):
        super(Conv1DBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, seq_len)

        Returns:
            output: (batch_size, out_channels, seq_len)
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out = out + residual
        out = self.relu(out)

        return out


class Spec2ClassCNN(nn.Module):
    """基于1D-CNN的质谱分类模型"""

    def __init__(self, num_classes: int, embedding_dim: int = 128,
                 hidden_dims: list = [256, 512, 256], dropout: float = 0.3):
        super(Spec2ClassCNN, self).__init__()

        self.peak_embedding = PeakEmbedding(embedding_dim)

        # 构建卷积层
        self.conv_blocks = nn.ModuleList()
        in_dim = embedding_dim
        for hidden_dim in hidden_dims:
            self.conv_blocks.append(Conv1DBlock(in_dim, hidden_dim, kernel_size=3, dropout=dropout))
            in_dim = hidden_dim

        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(in_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, mz_values: torch.Tensor, intensities: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            mz_values: (batch_size, num_peaks, 1)
            intensities: (batch_size, num_peaks, 1)
            mask: (batch_size, num_peaks)

        Returns:
            logits: (batch_size, num_classes)
        """
        # 峰嵌入
        x = self.peak_embedding(mz_values, intensities)  # (B, N, D)

        # 转置为卷积格式
        x = x.transpose(1, 2)  # (B, D, N)

        # 卷积特征提取
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # 全局池化
        avg_pool = self.global_avg_pool(x).squeeze(-1)  # (B, D)
        max_pool = self.global_max_pool(x).squeeze(-1)  # (B, D)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2D)

        # 分类
        logits = self.classifier(pooled)

        return logits


class Spec2ClassTransformer(nn.Module):
    """基于Transformer的质谱分类模型"""

    def __init__(self, num_classes: int, embedding_dim: int = 128,
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.3):
        super(Spec2ClassTransformer, self).__init__()

        self.peak_embedding = PeakEmbedding(embedding_dim)

        # Transformer编码器
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 前馈网络
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim * 4, embedding_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(embedding_dim)
            )
            for _ in range(num_layers)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, mz_values: torch.Tensor, intensities: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            mz_values: (batch_size, num_peaks, 1)
            intensities: (batch_size, num_peaks, 1)
            mask: (batch_size, num_peaks)

        Returns:
            logits: (batch_size, num_classes)
        """
        # 峰嵌入
        x = self.peak_embedding(mz_values, intensities)  # (B, N, D)

        # Transformer编码
        for attn_layer, ffn_layer in zip(self.attention_layers, self.ffn_layers):
            # Self-attention
            x = attn_layer(x, mask)

            # FFN with residual
            residual = x
            x = ffn_layer(x)
            x = x + residual

        # 池化（取第一个token或平均）
        if mask is not None:
            # 加权平均（考虑mask）
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = x.mean(dim=1)  # (B, D)

        # 分类
        logits = self.classifier(pooled)

        return logits


class Spec2ClassHybrid(nn.Module):
    """混合模型: CNN + Transformer (推荐) ⭐"""

    def __init__(self, num_classes: int, embedding_dim: int = 128,
                 cnn_channels: list = [256, 512], num_heads: int = 8,
                 num_attn_layers: int = 2, dropout: float = 0.3):
        super(Spec2ClassHybrid, self).__init__()

        self.peak_embedding = PeakEmbedding(embedding_dim)

        # CNN分支：提取局部特征
        self.conv_blocks = nn.ModuleList()
        in_dim = embedding_dim
        for out_dim in cnn_channels:
            self.conv_blocks.append(Conv1DBlock(in_dim, out_dim, kernel_size=5, dropout=dropout))
            in_dim = out_dim

        # 转换回序列格式
        self.cnn_to_seq = nn.Linear(in_dim, embedding_dim)

        # Transformer分支：捕捉全局依赖
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
            for _ in range(num_attn_layers)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, mz_values: torch.Tensor, intensities: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            mz_values: (batch_size, num_peaks, 1)
            intensities: (batch_size, num_peaks, 1)
            mask: (batch_size, num_peaks)

        Returns:
            logits: (batch_size, num_classes)
        """
        # 峰嵌入
        x = self.peak_embedding(mz_values, intensities)  # (B, N, D)

        # CNN分支
        cnn_x = x.transpose(1, 2)  # (B, D, N)
        for conv_block in self.conv_blocks:
            cnn_x = conv_block(cnn_x)
        cnn_x = cnn_x.transpose(1, 2)  # (B, N, D')
        cnn_x = self.cnn_to_seq(cnn_x)  # (B, N, D)

        # 融合CNN特征
        x = x + cnn_x  # 残差连接

        # Transformer分支
        for attn_layer in self.attention_layers:
            x = attn_layer(x, mask)

        # 池化
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = x.mean(dim=1)  # (B, D)

        # 分类
        logits = self.classifier(pooled)

        return logits


class Spec2ClassResNet(nn.Module):
    """基于ResNet的质谱分类模型"""

    def __init__(self, num_classes: int, embedding_dim: int = 128,
                 num_blocks: list = [2, 2, 2, 2], dropout: float = 0.3):
        super(Spec2ClassResNet, self).__init__()

        self.peak_embedding = PeakEmbedding(embedding_dim)

        # ResNet块
        dims = [embedding_dim, 128, 256, 512, 512]
        self.res_blocks = nn.ModuleList()

        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                in_dim = dims[i] if j == 0 else dims[i+1]
                self.res_blocks.append(
                    Conv1DBlock(in_dim, dims[i+1], kernel_size=3, dropout=dropout)
                )

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(dims[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, mz_values: torch.Tensor, intensities: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            mz_values: (batch_size, num_peaks, 1)
            intensities: (batch_size, num_peaks, 1)
            mask: (batch_size, num_peaks)

        Returns:
            logits: (batch_size, num_classes)
        """
        # 峰嵌入
        x = self.peak_embedding(mz_values, intensities)  # (B, N, D)
        x = x.transpose(1, 2)  # (B, D, N)

        # ResNet块
        for res_block in self.res_blocks:
            x = res_block(x)

        # 全局池化
        pooled = self.global_pool(x).squeeze(-1)  # (B, D)

        # 分类
        logits = self.classifier(pooled)

        return logits


def create_model(model_type: str, num_classes: int, **kwargs) -> nn.Module:
    """
    创建模型的工厂函数

    Args:
        model_type: 模型类型
            - 'cnn': 1D-CNN模型
            - 'transformer': Transformer模型
            - 'hybrid': CNN+Transformer混合模型（推荐）
            - 'resnet': ResNet模型
        num_classes: 分类类别数
        **kwargs: 其他模型参数

    Returns:
        model: PyTorch模型
    """
    model_dict = {
        'cnn': Spec2ClassCNN,
        'transformer': Spec2ClassTransformer,
        'hybrid': Spec2ClassHybrid,
        'resnet': Spec2ClassResNet
    }

    if model_type not in model_dict:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_dict.keys())}")

    model = model_dict[model_type](num_classes=num_classes, **kwargs)

    return model


if __name__ == '__main__':
    print("深度学习质谱分类模型库")
    print("\n可用模型:")
    print("  1. Spec2ClassCNN - 1D卷积神经网络")
    print("  2. Spec2ClassTransformer - Transformer编码器")
    print("  3. Spec2ClassHybrid - CNN+Transformer混合模型 ⭐ 推荐")
    print("  4. Spec2ClassResNet - 残差网络")

    # 测试
    batch_size = 4
    num_peaks = 100
    num_classes = 3

    # 模拟输入
    mz = torch.randn(batch_size, num_peaks, 1)
    intensity = torch.randn(batch_size, num_peaks, 1)
    mask = torch.ones(batch_size, num_peaks)

    print("\n测试模型:")
    for model_type in ['cnn', 'transformer', 'hybrid', 'resnet']:
        model = create_model(model_type, num_classes)
        with torch.no_grad():
            output = model(mz, intensity, mask)
        print(f"  {model_type}: 输入 ({batch_size}, {num_peaks}) -> 输出 {output.shape}")
