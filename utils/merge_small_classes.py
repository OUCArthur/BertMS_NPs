#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并少数类别工具

将样本数量过少的类别合并到相关类别或Others中，
以解决类别不平衡和过拟合问题
"""

import json
import argparse
from collections import Counter
from pathlib import Path


def merge_classes(input_path: str, output_path: str, min_samples: int = 20,
                 custom_merge_map: dict = None):
    """
    合并少数类别

    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        min_samples: 最小样本数阈值，小于此值的类别将被合并
        custom_merge_map: 自定义合并映射 {原类别: 新类别}
    """
    print(f"\n{'='*60}")
    print("合并少数类别工具")
    print(f"{'='*60}\n")

    # 加载数据
    print(f"加载数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[OK] 加载了 {len(data)} 个样本\n")

    # 统计原始类别分布
    original_classes = [item.get('detailed_class', item.get('library_class'))
                       for item in data]
    original_counts = Counter(original_classes)

    print("原始类别分布:")
    print("-" * 60)
    for cls, count in sorted(original_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(data) * 100
        mark = " <- 少数类别" if count < min_samples else ""
        print(f"{cls:<30} {count:>4} ({percentage:>5.1f}%){mark}")
    print()

    # 默认合并策略
    if custom_merge_map is None:
        merge_map = {
            # 合并脂质相关
            'Phospholipids': 'Lipids',
            'Fatty Acids/Lipids': 'Lipids',

            # 合并含氮化合物
            'Alkaloids': 'Nitrogenous Compounds',
            'Indole Derivatives': 'Nitrogenous Compounds',
        }

        # 将样本数<min_samples的类别合并到Others
        for cls, count in original_counts.items():
            if count < min_samples and cls not in merge_map and cls != 'Others':
                merge_map[cls] = 'Others'

    else:
        merge_map = custom_merge_map

    print("合并策略:")
    print("-" * 60)
    for old_cls, new_cls in sorted(merge_map.items()):
        old_count = original_counts.get(old_cls, 0)
        print(f"{old_cls:<30} -> {new_cls:<30} ({old_count} 样本)")
    print()

    # 应用合并
    merged_count = 0
    for item in data:
        old_class = item.get('detailed_class', item.get('library_class'))
        if old_class in merge_map:
            item['detailed_class'] = merge_map[old_class]
            merged_count += 1

    # 统计合并后的类别分布
    merged_classes = [item.get('detailed_class', item.get('library_class'))
                     for item in data]
    merged_counts = Counter(merged_classes)

    print("合并后的类别分布:")
    print("-" * 60)
    total_classes = len(merged_counts)
    for cls, count in sorted(merged_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(data) * 100
        print(f"{cls:<30} {count:>4} ({percentage:>5.1f}%)")

    print(f"\n合并统计:")
    print(f"  原始类别数: {len(original_counts)}")
    print(f"  合并后类别数: {len(merged_counts)}")
    print(f"  减少类别: {len(original_counts) - len(merged_counts)}")
    print(f"  受影响样本: {merged_count} ({merged_count/len(data)*100:.1f}%)")

    # 保存
    print(f"\n保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("[OK] 保存完成")

    # 生成报告
    report_path = output_path.replace('.json', '_merge_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("类别合并报告\n")
        f.write("="*60 + "\n\n")

        f.write(f"输入文件: {input_path}\n")
        f.write(f"输出文件: {output_path}\n")
        f.write(f"最小样本阈值: {min_samples}\n\n")

        f.write("合并策略:\n")
        f.write("-"*60 + "\n")
        for old_cls, new_cls in sorted(merge_map.items()):
            old_count = original_counts.get(old_cls, 0)
            f.write(f"{old_cls} ({old_count}样本) -> {new_cls}\n")

        f.write("\n合并前后对比:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'类别':<30} {'合并前':>10} {'合并后':>10} {'变化':>10}\n")
        f.write("-"*60 + "\n")

        all_classes = set(list(original_counts.keys()) + list(merged_counts.keys()))
        for cls in sorted(all_classes):
            before = original_counts.get(cls, 0)
            after = merged_counts.get(cls, 0)
            change = after - before
            change_str = f"+{change}" if change > 0 else str(change) if change < 0 else "0"
            f.write(f"{cls:<30} {before:>10} {after:>10} {change_str:>10}\n")

        f.write("\n" + "="*60 + "\n")
        f.write(f"原始类别数: {len(original_counts)}\n")
        f.write(f"合并后类别数: {len(merged_counts)}\n")
        f.write(f"总样本数: {len(data)}\n")

    print(f"[OK] 报告已保存到: {report_path}")
    print(f"\n{'='*60}")
    print("完成！")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='合并少数类别以解决类别不平衡问题')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入JSON文件路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='输出JSON文件路径')
    parser.add_argument('--min-samples', '-m', type=int, default=20,
                       help='最小样本数阈值（默认: 20）')

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return

    # 执行合并
    merge_classes(args.input, args.output, args.min_samples)


if __name__ == '__main__':
    # 示例用法
    print("\n" + "="*60)
    print("合并少数类别工具 - 示例")
    print("="*60)
    print("\n使用方法:")
    print("\npython utils/merge_small_classes.py \\")
    print("    --input data/ms/gnps_reclassified_1000.json \\")
    print("    --output data/ms/gnps_reclassified_1000_merged.json \\")
    print("    --min-samples 20")
    print("\n" + "="*60 + "\n")

    # 如果直接运行，使用默认参数
    input_file = "data/ms/gnps_reclassified_1000.json"
    output_file = "data/ms/gnps_reclassified_1000_merged.json"

    if Path(input_file).exists():
        print(f"发现数据文件: {input_file}")
        print("开始自动合并...\n")
        merge_classes(input_file, output_file, min_samples=20)
    else:
        print(f"数据文件不存在: {input_file}")
        print("请使用命令行参数指定输入文件")
