#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度学习质谱分类模型训练脚本 (改进版-防过拟合)

改进措施:
1. 添加标签平滑 (Label Smoothing)
2. 添加早停机制 (Early Stopping)
3. 提高Dropout率和Weight Decay
4. 调整默认超参数以防过拟合
5. 使用更简单的模型架构

Created by: BertMS Project
Date: 2025-01-24
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from models.spectrum_classifier import create_model
from utils.spectrum_dataset import create_dataloaders


class Trainer:
    """深度学习模型训练器（改进版）"""

    def __init__(self, model, train_loader, val_loader, config):
        """
        初始化训练器

        Args:
            model: PyTorch模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 配置字典
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 设备
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # 损失函数 - 添加标签平滑以防过拟合
        label_smoothing = config.get('label_smoothing', 0.1)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"使用标签平滑: {label_smoothing}")

        # 优化器
        learning_rate = config.get('lr', config.get('learning_rate', 0.001))
        weight_decay = config.get('weight_decay', 1e-3)

        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )

        # 学习率调度器
        if config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5,
                patience=5, verbose=True
            )
        elif config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config['epochs']
            )
        else:
            self.scheduler = None

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0

        # 早停机制
        self.patience = config.get('early_stopping_patience', 15)
        self.patience_counter = 0
        self.min_delta = config.get('early_stopping_min_delta', 0.001)
        print(f"早停机制: patience={self.patience}, min_delta={self.min_delta}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')

        for batch in pbar:
            # 数据移到设备
            mz = batch['mz'].to(self.device)
            intensity = batch['intensity'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(mz, intensity, mask)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.get('clip_grad', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['clip_grad']
                )

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Val]  ')

            for batch in pbar:
                # 数据移到设备
                mz = batch['mz'].to(self.device)
                intensity = batch['intensity'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                outputs = self.model(mz, intensity, mask)
                loss = self.criterion(outputs, labels)

                # 统计
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 保存预测结果
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy, all_preds, all_labels

    def train(self):
        """完整训练流程"""
        print("\n" + "="*70)
        print("开始训练 (改进版-防过拟合)")
        print("="*70)
        print(f"模型: {self.config.get('model', self.config.get('model_type', 'unknown'))}")
        print(f"设备: {self.device}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")
        print(f"批大小: {self.config['batch_size']}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Learning Rate: {self.config.get('lr', 0.001)}")
        print(f"Weight Decay: {self.config.get('weight_decay', 1e-3)}")
        print(f"Dropout: {self.config.get('dropout', 0.5)}")
        print("="*70 + "\n")

        for epoch in range(self.config['epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc, val_preds, val_labels = self.validate(epoch)

            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # 打印摘要
            print(f"\nEpoch {epoch+1}/{self.config['epochs']} 摘要:")
            print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
            print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
            print(f"  学习率: {current_lr:.6f}")

            # 计算准确率差距 (过拟合指标)
            acc_gap = train_acc - val_acc
            if acc_gap > 10:
                print(f"  ⚠️  准确率差距: {acc_gap:.2f}% (可能过拟合)")
            else:
                print(f"  ✓ 准确率差距: {acc_gap:.2f}%")

            # 保存最佳模型 & 早停检查
            if val_acc > self.best_val_acc + self.min_delta:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f"  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                print(f"  ⏳ 验证准确率未提升 ({self.patience_counter}/{self.patience})")

                if self.patience_counter >= self.patience:
                    print(f"\n⚠️  早停触发! 验证准确率连续{self.patience}个epoch未提升")
                    print(f"   最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
                    break

            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_acc)

            print("-" * 70)

        print("\n" + "="*70)
        print("训练完成!")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
        print("="*70 + "\n")

        return self.history

    def save_checkpoint(self, filename, epoch, val_acc):
        """保存检查点"""
        checkpoint_dir = Path(self.config['output_dir']) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config,
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_dir / filename)

    def save_final_model(self, dataset):
        """保存最终模型（用于推理）"""
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存完整模型
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.config['model_type'],
            'model_config': {
                'num_classes': self.config['num_classes'],
                'embedding_dim': self.config.get('embedding_dim', 64),
                'max_peaks': self.config['max_peaks']
            },
            'label_mapping': {
                'label_to_idx': dataset.label_to_idx,
                'idx_to_label': dataset.idx_to_label
            },
            'performance': {
                'best_val_acc': self.best_val_acc,
                'best_epoch': self.best_epoch
            },
            'history': self.history,
            'config': self.config
        }

        model_path = output_dir / 'spectrum_dl_classifier.pth'
        torch.save(model_data, model_path)
        print(f"✓ 模型已保存: {model_path}")

        # 保存训练历史
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ 训练历史已保存: {history_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练深度学习质谱分类模型 (改进版-防过拟合)')

    # 数据参数
    parser.add_argument('--data', type=str,
                       default='data/ms/gnps_all_cleand_by_matchms_and_pubchem_lookups_1000.json',
                       help='数据文件路径')
    parser.add_argument('--max-peaks', type=int, default=200,
                       help='最大峰数')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='训练集比例')

    # 模型参数 - 默认使用更简单的模型以防过拟合
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'transformer', 'hybrid', 'resnet'],
                       help='模型类型 (默认cnn以防过拟合)')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='嵌入维度 (降低以防过拟合)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout率 (提高以防过拟合)')

    # 训练参数 - 调整默认值以防过拟合
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批大小 (减小以增加正则化效果)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数 (配合早停使用)')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='学习率 (降低以防过拟合)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='权重衰减 (提高以防过拟合)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='标签平滑系数 (0表示不使用)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='优化器')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='学习率调度器')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                       help='梯度裁剪阈值（0表示不裁剪）')

    # 早停参数
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help='早停耐心值 (验证集未提升的epoch数)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.001,
                       help='早停最小改善阈值')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='数据加载线程数')
    parser.add_argument('--output-dir', type=str, default='results/dl_models_improved',
                       help='输出目录')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='保存检查点的频率（每N个epoch）')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config = vars(args)
    config['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')

    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("="*70)
    print("深度学习质谱分类模型训练 (改进版-防过拟合)")
    print("="*70)
    print("\n防过拟合措施:")
    print("  ✓ 标签平滑 (Label Smoothing)")
    print("  ✓ 早停机制 (Early Stopping)")
    print("  ✓ 更高的Dropout率")
    print("  ✓ 更强的Weight Decay")
    print("  ✓ 更小的模型架构")
    print("  ✓ 更小的批大小")
    print("  ✓ 更低的学习率")
    print()

    # 创建数据加载器
    train_loader, val_loader, dataset = create_dataloaders(
        data_path=args.data,
        batch_size=args.batch_size,
        train_split=args.train_split,
        max_peaks=args.max_peaks,
        num_workers=args.num_workers,
        random_seed=args.random_seed
    )

    # 更新配置
    config['num_classes'] = len(dataset.label_to_idx)
    config['model_type'] = args.model

    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = create_model(
        model_type=args.model,
        num_classes=config['num_classes'],
        embedding_dim=args.embedding_dim,
        dropout=args.dropout
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, config)

    # 训练
    history = trainer.train()

    # 保存最终模型
    trainer.save_final_model(dataset)

    print("\n所有任务完成!")
    print("\n💡 后续步骤:")
    print("  1. 运行诊断脚本检查是否还有过拟合:")
    print("     python diagnose_overfitting.py --model-path results/dl_models_improved/spectrum_dl_classifier.pth")
    print("  2. 如果仍然过拟合,考虑进一步减小模型或增加数据")


if __name__ == '__main__':
    main()
