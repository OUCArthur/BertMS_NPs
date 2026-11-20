"""
排序公平性分析脚本 — 对应审稿人批注6
验证 BertMS 的优势来自更好的排序能力，而非选择了更容易的样本。

分析内容：
  1. Rank 分析：固定同一批 query，对比 BertMS vs Cosine 对真实匹配的排名分布
     指标：Median Rank、Mean Reciprocal Rank (MRR)、Top-1/5/10/20 Recall
  2. 分数分布：正样本对与负样本对的相似度分布对比（两种方法）
  3. 两种方法在相同 query 集上的逐条排名散点图

用法：
  python document/eval_ranking_fairness.py \
      --model_path bert_base_ms/cache_xxx/model_ms_225.bin \
      --test_msp   data/ms/MoNA-export-LC-MS-MS_Positive_Mode_test.msp \
      --output_dir document/ranking_results
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
from matchms_ws.similarity.CosineGreedy import CosineGreedy
from matchms_ws import calculate_scores
from model import BertForMaskedLM_pre, BertConfig
from utils import LoadBertPretrainingDatasetMS


# ─── 预处理 ───────────────────────────────────────────────────────────────────
def preprocess(spectrum):
    spectrum = ms_filters.default_filters(spectrum)
    spectrum = ms_filters.normalize_intensities(spectrum)
    spectrum = ms_filters.select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = ms_filters.select_by_mz(spectrum, mz_from=10, mz_to=1000)
    return spectrum


# ─── 提取 BertMS embedding ────────────────────────────────────────────────────
def extract_embeddings(spectrums, model, config):
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
        for spectrum in tqdm(spectrums, desc="Extracting BertMS embeddings"):
            try:
                test_iter = data_loader.load_train_val_test_data(
                    test_file_path=spectrum,
                    train_file_path=None,
                    val_file_path=None,
                )
                for b_token_ids, b_mask, b_mlm_label in test_iter:
                    b_token_ids = b_token_ids.to(config.device)
                    b_mask      = b_mask.to(config.device)
                    b_mlm_label = b_mlm_label.to(config.device)
                    _, _, seq_out = model(
                        input_ids=b_token_ids,
                        attention_mask=b_mask,
                        masked_lm_labels=b_mlm_label,
                        token_type_ids=None,
                    )
                    intensities = torch.tensor(
                        spectrum.peaks.intensities, dtype=torch.float32, device=config.device
                    )
                    seq = seq_out.transpose(1, 0)
                    n_peaks = min(intensities.size(0), seq.size(1) - 2)
                    weights = torch.sqrt(intensities[:n_peaks])
                    peak_hidden = seq[0, 1:n_peaks + 1, :]
                    emb = (weights.unsqueeze(1) * peak_hidden).sum(dim=0)
                    all_embeddings.append(emb.cpu())
                    break
            except Exception:
                all_embeddings.append(torch.zeros(config.hidden_size))
    return torch.stack(all_embeddings, dim=0)  # [N, hidden]


# ─── 计算 Cosine Greedy 相似度矩阵 ───────────────────────────────────────────
def compute_cosine_matrix(spectrums):
    print("Computing CosineGreedy similarity matrix ...")
    cosine = CosineGreedy(tolerance=0.005)
    scores = calculate_scores(spectrums, spectrums, cosine, is_symmetric=True)
    arr = scores.scores.to_array()
    if hasattr(arr, 'cpu'):
        arr = arr.cpu().numpy()
    elif not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr.astype(np.float32)


# ─── InChIKey 标签 ────────────────────────────────────────────────────────────
def get_inchikeys(spectrums):
    return [( s.get("inchikey") or "")[:14] for s in spectrums]


# ─── Rank 指标计算 ────────────────────────────────────────────────────────────
def compute_rank_metrics(sim_matrix, inchikeys):
    """
    对每个 query，在排除自身后按相似度降序排列，
    找到第一个真实匹配（同 InChIKey 前14位）的排名（1-based）。
    只统计有真实匹配的 query。
    """
    n = len(inchikeys)
    ranks = []
    for i in range(n):
        if not inchikeys[i]:
            continue
        scores_i = sim_matrix[i].copy()
        scores_i[i] = -1  # 排除自身
        order = np.argsort(scores_i)[::-1]
        for rank, j in enumerate(order, start=1):
            if inchikeys[j] == inchikeys[i]:
                ranks.append(rank)
                break
    ranks = np.array(ranks)
    n_queries = len(ranks)
    metrics = {
        "n_queries":    n_queries,
        "median_rank":  int(np.median(ranks)),
        "mean_rank":    float(np.mean(ranks)),
        "MRR":          float(np.mean(1.0 / ranks)),
        "Top1":         float(np.mean(ranks <= 1)),
        "Top5":         float(np.mean(ranks <= 5)),
        "Top10":        float(np.mean(ranks <= 10)),
        "Top20":        float(np.mean(ranks <= 20)),
    }
    return metrics, ranks


# ─── 绘图函数 ─────────────────────────────────────────────────────────────────
def plot_rank_distributions(ranks_bert, ranks_cosine, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 累积排名曲线（越左越好）
    max_rank = min(200, max(ranks_bert.max(), ranks_cosine.max()))
    x = np.arange(1, max_rank + 1)
    cdf_bert   = np.array([np.mean(ranks_bert   <= k) for k in x])
    cdf_cosine = np.array([np.mean(ranks_cosine <= k) for k in x])

    axes[0].plot(x, cdf_bert,   lw=2,   label="BertMS")
    axes[0].plot(x, cdf_cosine, lw=2, linestyle="--", label="Cosine Greedy")
    axes[0].set_xlabel("Rank of first true match")
    axes[0].set_ylabel("Fraction of queries")
    axes[0].set_title("Cumulative Rank Distribution\n(same queries, same library)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 逐 query 排名散点图
    axes[1].scatter(ranks_cosine, ranks_bert, alpha=0.3, s=8, color="steelblue")
    lim = max(ranks_bert.max(), ranks_cosine.max())
    axes[1].plot([1, lim], [1, lim], "r--", lw=1, label="y = x")
    axes[1].set_xlabel("Cosine Greedy Rank")
    axes[1].set_ylabel("BertMS Rank")
    axes[1].set_title("Per-query Rank Comparison\n(points below diagonal: BertMS ranks better)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "rank_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Rank comparison saved → {path}")


def plot_score_distributions(bert_sim, cosine_sim, inchikeys, output_dir):
    n = len(inchikeys)
    rows, cols = np.triu_indices(n, k=1)
    labels = np.array(
        [1 if inchikeys[i] == inchikeys[j] and inchikeys[i] != "" else 0
         for i, j in zip(rows, cols)], dtype=np.int8
    )
    bert_scores   = bert_sim[rows, cols]
    cosine_scores = cosine_sim[rows, cols]

    pos_mask = labels == 1
    neg_mask = labels == 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.linspace(0, 1, 60)

    for ax, scores, title in zip(axes,
                                  [bert_scores, cosine_scores],
                                  ["BertMS", "Cosine Greedy"]):
        ax.hist(scores[neg_mask], bins=bins, alpha=0.6, density=True,
                label="Negative pairs (different compound)", color="steelblue")
        ax.hist(scores[pos_mask], bins=bins, alpha=0.7, density=True,
                label="Positive pairs (same compound)", color="tomato")
        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Density")
        ax.set_title(f"{title} Score Distribution")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "score_distributions.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Score distributions saved → {path}")


# ─── 主函数 ───────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载数据
    print("Loading spectra ...")
    spectrums_raw = list(load_from_msp(args.test_msp))
    spectrums = [preprocess(s) for s in spectrums_raw]
    spectrums = [s for s in spectrums
                 if s is not None and s.peaks.mz.size > 0 and s.get("inchikey")]
    print(f"  {len(spectrums)} spectra after filtering")

    inchikeys = get_inchikeys(spectrums)

    # 2. 加载 BertMS 模型
    print("Loading BertMS model ...")
    pretrained_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bert_base_ms"
    )
    config = BertConfig.from_json_file(os.path.join(pretrained_dir, "config.json"))
    config.vocab_path                  = os.path.join(pretrained_dir, "vocab.txt")
    config.device                      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.batch_size                  = 64
    config.max_sen_len                 = None
    config.pad_index                   = 0
    config.random_state                = 12022
    config.data_name                   = "ms"
    config.masked_rate                 = 0
    config.masked_token_rate           = 0
    config.masked_token_unchanged_rate = 0
    config.n_decimals                  = 2
    config.use_embedding_weight        = True
    config.use_torch_multi_head        = True
    config.is_sample_shuffle           = False

    model = BertForMaskedLM_pre.from_pretrained(config, model_path=args.model_path)
    model = model.to(config.device)

    # 3. BertMS 相似度矩阵
    print("Computing BertMS similarity matrix ...")
    embeddings = extract_embeddings(spectrums, model, config)
    normed = F.normalize(embeddings, dim=1)
    bert_sim = torch.mm(normed, normed.T).numpy()

    # 4. Cosine Greedy 相似度矩阵（相同谱图集）
    cosine_sim = compute_cosine_matrix(spectrums)

    # 5. Rank 指标
    print("\nComputing rank metrics (BertMS) ...")
    metrics_bert, ranks_bert = compute_rank_metrics(bert_sim, inchikeys)

    print("Computing rank metrics (Cosine Greedy) ...")
    metrics_cosine, ranks_cosine = compute_rank_metrics(cosine_sim, inchikeys)

    # 确保两者 query 集相同（取交集）
    assert len(ranks_bert) == len(ranks_cosine), \
        "Query counts differ — check InChIKey filtering"

    # 6. 打印结果
    header = f"{'Metric':<20} {'BertMS':>10} {'Cosine Greedy':>15}"
    print("\n" + header)
    print("-" * len(header))
    for key in ["n_queries", "median_rank", "mean_rank", "MRR",
                "Top1", "Top5", "Top10", "Top20"]:
        b = metrics_bert[key]
        c = metrics_cosine[key]
        fmt = ".4f" if isinstance(b, float) and b < 100 else "d" if isinstance(b, int) else ".1f"
        print(f"{key:<20} {b:>10.{fmt[-1]}} {c:>15.{fmt[-1]}}"
              if fmt[-1] in ('f',) else f"{key:<20} {b:>10} {c:>15}")

    # 7. 保存数值结果
    result_path = os.path.join(args.output_dir, "ranking_results.txt")
    with open(result_path, "w") as f:
        f.write(f"Test set : {args.test_msp}\n")
        f.write(f"Model    : {args.model_path}\n")
        f.write(f"N spectra: {len(spectrums)}\n\n")
        f.write(f"{'Metric':<20} {'BertMS':>10} {'Cosine Greedy':>15}\n")
        f.write("-" * 47 + "\n")
        for key in ["n_queries", "median_rank", "mean_rank", "MRR",
                    "Top1", "Top5", "Top10", "Top20"]:
            b, c = metrics_bert[key], metrics_cosine[key]
            f.write(f"{key:<20} {b:>10} {c:>15}\n")
    print(f"\n  Results saved → {result_path}")

    # 8. 绘图
    plot_rank_distributions(ranks_bert, ranks_cosine, args.output_dir)
    plot_score_distributions(bert_sim, cosine_sim, inchikeys, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="BertMS 模型文件路径")
    parser.add_argument("--test_msp",   required=True,
                        help="测试集 MSP 文件路径")
    parser.add_argument("--output_dir", default="document/ranking_results",
                        help="结果输出目录")
    args = parser.parse_args()
    main(args)
