"""
消融实验训练脚本：验证三种 embedding 组合对 BertMS 性能的影响。

实验组：
  full        — token + position + segment  (完整模型)
  no_segment  — token + position
  no_position — token + segment
  token_only  — token only

使用方法：
  python train_ablation.py --variant full
  python train_ablation.py --variant no_segment
  python train_ablation.py --variant no_position
  python train_ablation.py --variant token_only

不修改原有任何代码，通过 monkey-patch 替换 bert_embeddings 层来实现消融。
"""

import sys
import os
import argparse
import logging
import time
import types
import torch
import torch.nn.functional as F
from copy import deepcopy
from datetime import datetime

# 把项目根目录加入 path，保持与原脚本一致
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from transformers import BertTokenizer
from transformers import get_polynomial_decay_schedule_with_warmup

from model import BertForMaskedLM, BertConfig
from utils import LoadBertPretrainingDatasetMS, logger_init
from bert_embedding_ablation import BertEmbeddingsAblation


# ─── 消融配置表 ────────────────────────────────────────────────────────────────
ABLATION_VARIANTS = {
    "full":        {"use_position": True,  "use_segment": True},
    "no_segment":  {"use_position": True,  "use_segment": False},
    "no_position": {"use_position": False, "use_segment": True},
    "token_only":  {"use_position": False, "use_segment": False},
}


# ─── 模型 + 训练配置 ───────────────────────────────────────────────────────────
class ModelConfig:
    def __init__(self, variant: str):
        assert variant in ABLATION_VARIANTS, f"variant 必须是 {list(ABLATION_VARIANTS.keys())} 之一"
        self.variant = variant

        self.project_dir = PROJECT_DIR
        self.dataset_dir = os.path.join(self.project_dir, "data", "ms")
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_ms")

        self.train_file_path = os.path.join(
            self.dataset_dir,
            "gnps_all_cleand_by_matchms_and_pubchem_lookups_62081_s2v_qian_no1000.msp"
        )
        self.val_file_path = os.path.join(
            self.dataset_dir, "MoNA-export-LC-MS-MS_Positive_Mode_val.msp"
        )
        self.test_file_path = os.path.join(
            self.dataset_dir, "MoNA-export-LC-MS-MS_Positive_Mode_test.msp"
        )
        self.data_name = "ms"

        self.vocab_path = os.path.join(self.pretrained_model_dir, "vocab.txt")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 每个 variant 保存到独立目录，便于对比
        self.model_save_dir = os.path.join(
            self.pretrained_model_dir,
            f"ablation_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.logs_save_dir = os.path.join(self.pretrained_model_dir, "logs")
        self.model_save_path = os.path.join(self.model_save_dir, f"model_{self.data_name}.bin")

        # 预训练权重路径（用于消融实验的初始化，避免从随机权重训练导致不稳定）
        self.pretrained_weights_path = os.path.join(
            self.pretrained_model_dir,
            "cache_20240325_215535",
            "model_ms_225.bin"
        )

        # 消融开关（从 ABLATION_VARIANTS 读取）
        self.use_position_embedding = ABLATION_VARIANTS[variant]["use_position"]
        self.use_segment_embedding = ABLATION_VARIANTS[variant]["use_segment"]

        # 训练超参（与原脚本保持一致）
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 8
        self.max_sen_len = None
        self.grad_accumulation_steps = 8   # 等效 batch_size = 8 × 8 = 64
        self.pad_index = 0
        self.random_state = 12022
        self.learning_rate = 1e-5
        self.weight_decay = 0.1
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.log_level = logging.DEBUG
        self.use_torch_multi_head = True
        self.epochs = 3000
        self.model_val_per_epoch = 5
        self.n_decimals = 2

        logger_init(
            log_file_name=f"{self.data_name}_ablation_{variant}",
            log_level=self.log_level,
            log_dir=self.logs_save_dir,
        )
        os.makedirs(self.model_save_dir, exist_ok=True)

        bert_config = BertConfig.from_json_file(
            os.path.join(self.pretrained_model_dir, "config.json")
        )
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value

        logging.info(f"### 消融实验 variant = {variant}")
        logging.info(f"### use_position_embedding = {self.use_position_embedding}")
        logging.info(f"### use_segment_embedding  = {self.use_segment_embedding}")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────
def accuracy_sparse(logits_masked, labels_masked):
    """仅计算 mask 位置的准确率，避免依赖完整 vocab 投影张量。

    logits_masked: [M, vocab_size]
    labels_masked: [M]
    """
    M = labels_masked.shape[0]
    if M == 0:
        return 0.0, 0, 0
    pred = logits_masked.argmax(dim=-1)
    correct = pred.eq(labels_masked).sum().item()
    return float(correct) / M, correct, M


def evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    mlm_corrects, mlm_totals = 0, 0
    last_loss = torch.tensor(float('nan'))
    with torch.no_grad():
        for b_token_ids, b_mask, b_mlm_label in data_iter:
            b_token_ids = b_token_ids.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            result = model(
                    input_ids=b_token_ids,
                    attention_mask=b_mask,
                    masked_lm_labels=b_mlm_label,
                    token_type_ids=None,
                )
            if result[0] is None:
                continue
            logits_masked, last_loss = result
            labels_flat = b_mlm_label.reshape(-1)
            mask_pos = labels_flat != PAD_IDX
            _, mlm_cor, mlm_tot = accuracy_sparse(logits_masked, labels_flat[mask_pos])
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
    model.train()
    acc = float(mlm_corrects) / mlm_totals if mlm_totals > 0 else 0.0
    return acc, last_loss


# ─── 构建模型（含 monkey-patch）─────────────────────────────────────────────
def _sparse_forward(model_self, input_ids, attention_mask=None,
                    token_type_ids=None, position_ids=None, masked_lm_labels=None):
    """替换 BertForMaskedLM.forward：只对 mask 位置做 vocab 投影。

    vocab_size=94891 导致完整投影张量 [S, B, V] 极大（~5 GiB），
    这里改为先提取 mask 位置的 hidden state，再投影到 vocab，
    只需分配 [num_masked, vocab_size]（通常 <100 MB）。

    返回：
      (logits_masked [M, V], loss)  —— masked_lm_labels 不为 None 且有 mask token
      (None, None)                  —— 该 batch 无 mask token，跳过
    """
    _, all_encoder_outputs = model_self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
    )
    seq_out = all_encoder_outputs[-1]          # [S, B, H]

    labels_flat = masked_lm_labels.reshape(-1) if masked_lm_labels is not None else None
    mask_pos = (labels_flat != 0) if labels_flat is not None else None

    # 一次性诊断日志，确认 label 张量格式
    if not hasattr(model_self, '_debug_logged'):
        model_self._debug_logged = True
        if labels_flat is not None:
            logging.info(
                f"## DEBUG labels: shape={masked_lm_labels.shape}, "
                f"dtype={masked_lm_labels.dtype}, "
                f"nonzero={labels_flat.count_nonzero().item()}, "
                f"min={labels_flat.min().item()}, max={labels_flat.max().item()}"
            )

    if mask_pos is not None and not mask_pos.any():
        return None, None

    # 只取 mask 位置的 hidden state，再过 LM head（dense → act → LN → decoder）
    hs = seq_out.reshape(-1, seq_out.shape[-1])        # [S*B, H]
    if mask_pos is not None:
        hs = hs[mask_pos]                              # [M, H]

    cls = model_self.classifier
    hs = cls.dense(hs)
    if cls.transform_act_fn is not None:
        hs = cls.transform_act_fn(hs)
    hs = cls.LayerNorm(hs)
    logits = cls.decoder(hs)                           # [M, V]

    if mask_pos is not None:
        loss = F.cross_entropy(logits, labels_flat[mask_pos])
        return logits, loss
    else:
        return logits  # 纯推理路径（本实验不走这里）


def build_model(config):
    """
    创建 BertForMaskedLM，然后用消融版 embedding 替换 bert_embeddings，
    并 monkey-patch forward 以避免分配完整 [S, B, vocab_size] 张量。
    从预训练权重初始化以保证训练稳定性。
    """
    model = BertForMaskedLM(config, None)

    # ── 替换 embedding 层 ──
    ablation_embedding = BertEmbeddingsAblation(
        config,
        use_position=config.use_position_embedding,
        use_segment=config.use_segment_embedding,
    )
    model.bert.bert_embeddings = ablation_embedding

    # ── 加载预训练权重（解决随机初始化导致的训练不稳定/NaN问题）──
    pretrained_path = getattr(config, 'pretrained_weights_path', None)
    if pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state = checkpoint.get('model_state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logging.info(
            f"## 预训练权重载入: {pretrained_path}\n"
            f"   missing({len(missing)}): {missing[:3]}...\n"
            f"   unexpected({len(unexpected)}): {unexpected[:3]}..."
        )
    else:
        logging.warning(
            f"## 预训练权重文件不存在或未配置: {pretrained_path}，"
            f"将使用随机初始化（训练可能不稳定）"
        )

    # ── 权重绑定：在 load_state_dict 之后操作 _parameters 以确保真正共享 ──
    # 用 __setattr__ 赋值会走 nn.Module 的注册逻辑，无法保证共享同一对象；
    # 直接操作 _parameters 才能让两条路径指向同一 Parameter。
    if config.use_embedding_weight:
        shared_param = model.bert.bert_embeddings.word_embeddings.embedding.weight
        model.classifier.decoder._parameters['weight'] = shared_param
        logging.info(f"## 权重绑定完成，形状: {shared_param.shape}")

    # ── monkey-patch forward：只对 mask 位置做 vocab 投影 ──
    model.forward = types.MethodType(_sparse_forward, model)
    logging.info("## forward 已替换为 sparse 版本（只投影 mask 位置）")

    logging.info(
        f"## 消融配置: use_position={config.use_position_embedding}, "
        f"use_segment={config.use_segment_embedding}"
    )
    return model


# ─── 训练主函数 ───────────────────────────────────────────────────────────────
def train(config):
    model = build_model(config)

    last_epoch = -1
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        last_epoch = checkpoint["last_epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info("## 成功载入已有模型，进行追加训练......")

    model = model.to(config.device)
    model.train()

    # ── 启动诊断：确认参数去重、检查初始权重是否含 NaN ──
    all_params = list(model.parameters())
    unique_ids = {id(p) for p in all_params}
    logging.info(f"## 参数总数: {len(all_params)}, 去重后: {len(unique_ids)}")
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logging.error(f"## 初始权重异常: {name}, nan={torch.isnan(param).sum()}, inf={torch.isinf(param).sum()}")
    emb_id = id(model.bert.bert_embeddings.word_embeddings.embedding.weight)
    dec_id = id(model.classifier.decoder.weight)
    logging.info(f"## 权重绑定去重验证: emb_id={emb_id}, dec_id={dec_id}, 相同={emb_id == dec_id}")

    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDatasetMS(
        vocab_path=config.vocab_path,
        tokenizer=bert_tokenize,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_index,
        is_sample_shuffle=config.is_sample_shuffle,
        random_state=config.random_state,
        data_name=config.data_name,
        masked_rate=config.masked_rate,
        masked_token_rate=config.masked_token_rate,
        masked_token_unchanged_rate=config.masked_token_unchanged_rate,
        n_decimals=config.n_decimals,
    )
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(
        test_file_path=config.test_file_path,
        train_file_path=config.train_file_path,
        val_file_path=config.val_file_path,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # warmup: 5 个 epoch 等效的 optimizer step 数（每 grad_accumulation_steps 个 batch 走一步）
    warmup_steps = int(len(train_iter) / config.grad_accumulation_steps * 5)
    total_steps = int(config.epochs * len(train_iter) / config.grad_accumulation_steps)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
        last_epoch=last_epoch,
    )

    max_acc = 0
    state_dict = None

    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()

        optimizer.zero_grad()
        has_backward = False
        valid_batches = 0
        for idx, (b_token_ids, b_mask, b_mlm_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)

            result = model(
                    input_ids=b_token_ids,
                    attention_mask=b_mask,
                    token_type_ids=None,
                    masked_lm_labels=b_mlm_label,
                )

            if result[0] is None:
                # 该 batch 无 mask token，跳过 backward
                masked_lm_loss = None
                mlm_logits_masked = None
            else:
                mlm_logits_masked, masked_lm_loss = result

            if masked_lm_loss is not None:
                if torch.isnan(masked_lm_loss) or torch.isinf(masked_lm_loss):
                    logging.warning(
                        f"[{config.variant}] NaN/Inf loss at epoch {epoch+1} batch {idx}, skipping backward"
                    )
                else:
                    (masked_lm_loss / config.grad_accumulation_steps).backward()
                    losses += masked_lm_loss.item()
                    valid_batches += 1
                    has_backward = True

            if (idx + 1) % config.grad_accumulation_steps == 0 or (idx + 1) == len(train_iter):
                if has_backward:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logging.warning(
                            f"[{config.variant}] NaN/Inf gradient norm at epoch {epoch+1} batch {idx}, "
                            f"skipping optimizer step"
                        )
                    else:
                        optimizer.step()
                        scheduler.step()
                    has_backward = False
                optimizer.zero_grad()

            log_every = 1 if epoch == 0 else 20  # 第一个 epoch 每 batch 都打，后续每 20 个 batch
            if idx % log_every == 0:
                if mlm_logits_masked is not None:
                    labels_flat = b_mlm_label.reshape(-1)
                    mask_pos = labels_flat != data_loader.PAD_IDX
                    mlm_acc, _, _ = accuracy_sparse(
                        mlm_logits_masked.detach(), labels_flat[mask_pos]
                    )
                    loss_val = masked_lm_loss.item()
                else:
                    mlm_acc, loss_val = 0.0, float('nan')
                logging.info(
                    f"[{config.variant}] Epoch [{epoch+1}/{config.epochs}], "
                    f"Batch [{idx}/{len(train_iter)}], "
                    f"Loss: {loss_val:.3f}, MLM Acc: {mlm_acc:.3f}"
                )

        train_loss = losses / valid_batches if valid_batches > 0 else float('nan')
        elapsed = time.time() - start_time
        logging.info(
            f"[{config.variant}] Epoch [{epoch+1}/{config.epochs}], "
            f"Train Loss: {train_loss:.3f}, Time: {elapsed:.1f}s"
        )

        if (epoch + 1) % config.model_val_per_epoch == 0:
            mlm_acc, val_loss = evaluate(config, val_iter, model, data_loader.PAD_IDX)
            logging.info(
                f"[{config.variant}] Val MLM Acc: {mlm_acc:.4f}, Val Loss: {val_loss:.3f}"
            )
            if mlm_acc > max_acc:
                max_acc = mlm_acc
                state_dict = deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 and state_dict is not None:
            ckpt_path = os.path.join(
                config.model_save_dir, f"model_{config.data_name}_{epoch+1}.bin"
            )
            torch.save({"last_epoch": scheduler.last_epoch, "model_state_dict": state_dict}, ckpt_path)
            logging.info(f"[{config.variant}] 已保存: {ckpt_path}")

    if state_dict is not None:
        torch.save(
            {"last_epoch": scheduler.last_epoch, "model_state_dict": state_dict},
            config.model_save_path,
        )
        logging.info(f"[{config.variant}] 最终模型已保存: {config.model_save_path}")


# ─── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BertMS Embedding 消融实验")
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=list(ABLATION_VARIANTS.keys()),
        help="消融实验组: full | no_segment | no_position | token_only",
    )
    args = parser.parse_args()

    config = ModelConfig(variant=args.variant)
    train(config)
