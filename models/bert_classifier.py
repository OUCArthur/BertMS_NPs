#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于BERT向量的质谱分类器
使用预训练的BERT模型提取质谱特征，然后训练分类头

方案1：冻结BERT + 轻量分类头
- BERT模型权重冻结，不参与训练
- 只训练顶层的MLP分类器
- 适合数据量较小的情况

Created by: BertMS Project
Date: 2025-01-24
"""

import torch
import torch.nn as nn
from typing import Optional


class BertSpectrumClassifier(nn.Module):
    """
    基于BERT向量的质谱分类器

    架构：
    质谱 -> BERT编码器(冻结) -> [CLS] token -> MLP分类头 -> 类别概率
    """

    def __init__(
        self,
        bert_model,
        num_classes: int,
        hidden_size: int = 768,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_bert: bool = True
    ):
        """
        初始化BERT分类器

        Args:
            bert_model: 预训练的BERT模型（BertForMaskedLM或BertModel）
            num_classes: 分类类别数
            hidden_size: BERT隐藏层大小（默认768）
            classifier_hidden_dim: 分类器隐藏层大小
            dropout: Dropout比例
            freeze_bert: 是否冻结BERT参数（默认True，方案1）
        """
        super(BertSpectrumClassifier, self).__init__()

        self.bert_model = bert_model
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.freeze_bert = freeze_bert

        # 冻结BERT参数（方案1）
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            print("✓ BERT参数已冻结，只训练分类头")
        else:
            print("⚠ BERT参数可训练（端到端微调）")

        # MLP分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, num_classes)
        )

        # 初始化分类器权重
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """初始化分类器权重"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def extract_bert_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        使用BERT提取质谱特征

        Args:
            input_ids: [src_len, batch_size] 或 [batch_size, src_len]
            attention_mask: [batch_size, src_len]
            token_type_ids: [src_len, batch_size] 或 [batch_size, src_len]

        Returns:
            features: [batch_size, hidden_size] BERT提取的特征向量
        """
        # 检查BERT模型是否有bert属性（BertForMaskedLM等）
        if hasattr(self.bert_model, 'bert'):
            bert = self.bert_model.bert
        else:
            bert = self.bert_model

        # 前向传播获取BERT输出
        with torch.set_grad_enabled(not self.freeze_bert):
            pooled_output, sequence_output = bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            # pooled_output: [batch_size, hidden_size] - [CLS] token的表示
            # sequence_output: [src_len, batch_size, hidden_size] - 所有token的表示

        return pooled_output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: token序列
            attention_mask: 注意力mask
            token_type_ids: token类型ID

        Returns:
            logits: [batch_size, num_classes] 分类logits
        """
        # 1. 提取BERT特征
        features = self.extract_bert_features(input_ids, attention_mask, token_type_ids)

        # 2. 分类
        logits = self.classifier(features)

        return logits

    def get_feature_vector(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取质谱的BERT特征向量（用于可视化、聚类等）

        Returns:
            features: [batch_size, hidden_size]
        """
        self.eval()
        with torch.no_grad():
            features = self.extract_bert_features(input_ids, attention_mask, token_type_ids)
        return features

    def unfreeze_bert(self):
        """解冻BERT参数（用于方案3：渐进式微调）"""
        for param in self.bert_model.parameters():
            param.requires_grad = True
        self.freeze_bert = False
        print("✓ BERT参数已解冻")

    def unfreeze_bert_top_layers(self, num_layers: int = 2):
        """
        只解冻BERT的顶层（用于方案3：渐进式微调）

        Args:
            num_layers: 解冻的层数
        """
        # 获取BERT编码器
        if hasattr(self.bert_model, 'bert'):
            bert = self.bert_model.bert
        else:
            bert = self.bert_model

        # 解冻顶层
        if hasattr(bert, 'encoder') and hasattr(bert.encoder, 'layers'):
            total_layers = len(bert.encoder.layers)
            for i in range(total_layers - num_layers, total_layers):
                for param in bert.encoder.layers[i].parameters():
                    param.requires_grad = True
            print(f"✓ 已解冻BERT顶部 {num_layers} 层")


def create_bert_classifier(
    bert_model,
    num_classes: int,
    hidden_size: int = 768,
    classifier_hidden_dim: int = 256,
    dropout: float = 0.3,
    freeze_bert: bool = True
) -> BertSpectrumClassifier:
    """
    便捷函数：创建BERT分类器

    Args:
        bert_model: 预训练的BERT模型
        num_classes: 分类类别数
        hidden_size: BERT隐藏层大小
        classifier_hidden_dim: 分类器隐藏层大小
        dropout: Dropout比例
        freeze_bert: 是否冻结BERT

    Returns:
        BertSpectrumClassifier实例
    """
    return BertSpectrumClassifier(
        bert_model=bert_model,
        num_classes=num_classes,
        hidden_size=hidden_size,
        classifier_hidden_dim=classifier_hidden_dim,
        dropout=dropout,
        freeze_bert=freeze_bert
    )
