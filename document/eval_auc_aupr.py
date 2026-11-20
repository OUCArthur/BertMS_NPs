"""
AUC / AUPR 评估脚本 — 对应审稿人批注5
计算 BertMS 在全局谱图相似度检索任务上的 ROC-AUC 和 PR-AUC（AUPR）。

定义：
  正样本对：InChIKey 前14位相同（同一化合物，不区分立体异构体）
  负样本对：InChIKey 前14位不同

用法（服务器上）：
  python document/eval_auc_aupr.py \
      --model_path bert_base_ms/cache_xxx/model_ms_225.bin \
      --test_msp   data/ms/MoNA-export-LC-MS-MS_Positive_Mode_test.msp \
      --output_dir document/auc_results
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matchms_ws.filtering as ms_filters
from matchms_ws.importing import load_from_msp
from model import BertForMaskedLM_pre, BertConfig
from utils import LoadBertPretrainingDatasetMS
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)


# ─── 预处理 ───────────────────────────────────────────────────────────────────
def preprocess(spectrum):
    spectrum = ms_filters.default_filters(spectrum)
    spectrum = ms_filters.normalize_intensities(spectrum)
    spectrum = ms_filters.select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = ms_filters.select_by_mz(spectrum, mz_from=10, mz_to=1000)
    return spectrum


# ─── 提取所有谱图的 embedding ──────────────────────────────────────────────────
def extract_embeddings(spectrums, model, config):
    """
    对每条谱图提取 intensity-weighted hidden state sum，返回 [N, hidden_size]。
    """
    data_loader = LoadBertPretrainingDatasetMS(
        vocab_path=config.vocab_path,
        tokenizer=None,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_index,
        is_sample_shuffle=False,
        random_state=config.random_state,
        data_name=config.data_name,
        masked_rate=0,
        masked_token_rate=0,
        masked_token_unchanged_rate=0,
        n_decimals=config.n_decimals,
    )

    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for spectrum in tqdm(spectrums, desc="Extracting embeddings"):
            try:
                test_iter = data_loader.load_train_val_test_data(
                    test_file_path=spectrum,
                    train_file_path=None,
                    val_file_path=None,
                )
                for b_token_ids, b_mask, b_mlm_label in test_iter:
                    b_token_ids = b_token_ids.to(config.device)
                    b_mask = b_mask.to(config.device)
                    b_mlm_label = b_mlm_label.to(config.device)

                    _, _, sequence_output = model(
                        input_ids=b_token_ids,
                        attention_mask=b_mask,
                        masked_lm_labels=b_mlm_label,
                        token_type_ids=None,
                    )
                    # intensity-weighted sum: sqrt(I_i) * h_i
                    intensities = torch.tensor(
                        spectrum.peaks.intensities, dtype=torch.float32, device=config.device
                    )
                    seq = sequence_output.transpose(1, 0)  # [batch, src_len, hidden]
                    n_peaks = min(intensities.size(0), seq.size(1) - 2)  # exclude CLS/SEP
                    weights = torch.sqrt(intensities[:n_peaks])           # [n_peaks]
                    peak_hidden = seq[0, 1:n_peaks + 1, :]               # [n_peaks, hidden]
                    emb = (weights.unsqueeze(1) * peak_hidden).sum(dim=0) # [hidden]
                    all_embeddings.append(emb.cpu())
                    break  # single spectrum per iter
            except Exception:
                # 跳过无法处理的谱图，用零向量占位
                all_embeddings.append(torch.zeros(config.hidden_size))

    return torch.stack(all_embeddings, dim=0)  # [N, hidden_size]


# ─── 构造真实标签矩阵 ─────────────────────────────────────────────────────────
def build_labels(spectrums):
    """
    返回上三角的 (scores_flat, labels_flat)，用于 AUC 计算。
    正样本：InChIKey 前14位相同；负样本：不同。
    """
    inchikeys = []
    for s in spectrums:
        ik = s.get("inchikey") or ""
        inchikeys.append(ik[:14])   # 前14位 = 分子连接层，不区分立体异构

    n = len(inchikeys)
    rows, cols = np.triu_indices(n, k=1)
    labels = np.array(
        [1 if inchikeys[i] == inchikeys[j] and inchikeys[i] != "" else 0
         for i, j in zip(rows, cols)],
        dtype=np.int8,
    )
    return rows, cols, labels


# ─── 计算余弦相似度（分块，节省显存）────────────────────────────────────────
def cosine_similarity_matrix(embeddings, chunk_size=512):
    """返回归一化后的余弦相似度矩阵 [N, N]（CPU numpy）。"""
    normed = F.normalize(embeddings, dim=1)  # [N, hidden]
    n = normed.size(0)
    sim = torch.zeros(n, n)
    for i in tqdm(range(0, n, chunk_size), desc="Computing similarity"):
        chunk = normed[i:i + chunk_size]
        sim[i:i + chunk_size] = torch.mm(chunk, normed.T).cpu()
    return sim.numpy()


# ─── 主函数 ───────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载数据
    print("Loading spectra ...")
    spectrums_raw = list(load_from_msp(args.test_msp))
    spectrums = [preprocess(s) for s in spectrums_raw]
    spectrums = [s for s in spectrums if s is not None and s.peaks.mz.size > 0 and s.get("inchikey")]
    print(f"  {len(spectrums)} spectra after filtering")

    # 2. 加载模型
    print("Loading model ...")
    pretrained_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bert_base_ms")
    config = BertConfig.from_json_file(os.path.join(pretrained_dir, "config.json"))
    config.vocab_path      = os.path.join(pretrained_dir, "vocab.txt")
    config.device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.batch_size      = 64
    config.max_sen_len     = None
    config.pad_index       = 0
    config.random_state    = 12022
    config.data_name       = "ms"
    config.masked_rate     = 0
    config.masked_token_rate      = 0
    config.masked_token_unchanged_rate = 0
    config.n_decimals      = 2
    config.use_embedding_weight   = True
    config.use_torch_multi_head   = True
    config.is_sample_shuffle      = False

    model = BertForMaskedLM_pre.from_pretrained(config, model_path=args.model_path)
    model = model.to(config.device)
    model.eval()

    # 3. 提取 embedding
    embeddings = extract_embeddings(spectrums, model, config)
    print(f"  Embeddings shape: {embeddings.shape}")

    # 4. 计算相似度矩阵 & 真实标签
    print("Computing similarity matrix ...")
    sim_matrix = cosine_similarity_matrix(embeddings)

    print("Building labels ...")
    rows, cols, labels = build_labels(spectrums)
    scores_flat = sim_matrix[rows, cols]

    pos = labels.sum()
    neg = len(labels) - pos
    print(f"  Positive pairs: {pos:,}  |  Negative pairs: {neg:,}")

    # 5. 计算指标
    roc_auc  = roc_auc_score(labels, scores_flat)
    pr_auc   = average_precision_score(labels, scores_flat)
    print(f"\n  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    # 6. 保存数值结果
    result_path = os.path.join(args.output_dir, "auc_results.txt")
    with open(result_path, "w") as f:
        f.write(f"Test set : {args.test_msp}\n")
        f.write(f"Model    : {args.model_path}\n")
        f.write(f"N spectra: {len(spectrums)}\n")
        f.write(f"Pos pairs: {pos}\n")
        f.write(f"Neg pairs: {neg}\n")
        f.write(f"ROC-AUC  : {roc_auc:.4f}\n")
        f.write(f"PR-AUC   : {pr_auc:.4f}\n")
    print(f"  Results saved → {result_path}")

    # 7. 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(labels, scores_flat)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=1.5, label=f"BertMS (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — BertMS Spectral Similarity")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"  ROC curve saved → {roc_path}")

    # 8. 绘制 PR 曲线
    precision, recall, _ = precision_recall_curve(labels, scores_flat)
    baseline = pos / len(labels)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, lw=1.5, label=f"BertMS (AUPR = {pr_auc:.3f})")
    plt.axhline(y=baseline, color="k", linestyle="--", lw=0.8,
                label=f"Random baseline ({baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve — BertMS Spectral Similarity")
    plt.legend(loc="upper right")
    plt.tight_layout()
    pr_path = os.path.join(args.output_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"  PR curve saved → {pr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="训练好的模型文件路径，如 bert_base_ms/cache_xxx/model_ms_225.bin")
    parser.add_argument("--test_msp", required=True,
                        help="测试集 MSP 文件路径")
    parser.add_argument("--output_dir", default="document/auc_results",
                        help="结果输出目录")
    args = parser.parse_args()
    main(args)
