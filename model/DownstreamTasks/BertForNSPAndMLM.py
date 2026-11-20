import logging
from ..BasicBert.Bert import BertModel
from ..BasicBert.Bert import get_activation
import torch.nn as nn
import torch


class BertForNextSentencePrediction(nn.Module):
    """
    仅为下一句预测模型
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForNextSentencePrediction, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self,
                input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size] 如果输入模型的只有一个序列，那么这个参数也不用传值
                position_ids=None,
                next_sentence_labels=None  # [batch_size,]
                ):
        pooled_output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        # pooled_output: [batch_size, hidden_size]
        seq_relationship_score = self.classifier(pooled_output)
        # seq_relationship_score: [batch_size, 2]
        if next_sentence_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
            return loss
        else:
            return seq_relationship_score


class BertForLMTransformHead(nn.Module):
    """
    用于BertForMaskedLM中的一次变换。 因为在单独的MLM任务中
    和最后NSP与MLM的整体任务中均要用到，所以这里单独抽象为一个类便于复用

    ref: https://github.com/google-research/bert/blob/master/run_pretraining.py
        第248-262行
    """

    def __init__(self, config, bert_model_embedding_weights=None):
        """
        :param config:
        :param bert_model_embedding_weights:
        the output-weights are the same as the input embeddings, but there is
        an output-only bias for each token. 即TokenEmbedding层中的词表矩阵
        """
        super(BertForLMTransformHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        if bert_model_embedding_weights is not None:
            self.decoder.weight = nn.Parameter(bert_model_embedding_weights)
        # [hidden_size, vocab_size]
        self.decoder.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """
        :param hidden_states: [src_len, batch_size, hidden_size] Bert最后一层的输出
        :return:
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.transform_act_fn(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.LayerNorm(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.decoder(hidden_states)
        # hidden_states:  [src_len, batch_size, vocab_size]
        return hidden_states


class BertForMaskedLM(nn.Module):
    """
    仅为掩码语言预测模型
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForMaskedLM, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        weights = None
        if config.use_embedding_weight:
            weights = self.bert.bert_embeddings.word_embeddings.embedding.weight
            logging.info(f"## 使用token embedding中的权重矩阵作为输出层的权重！{weights.shape}")
        self.classifier = BertForLMTransformHead(config, weights)
        self.config = config

    def forward(self,
                input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size]
                position_ids=None,
                masked_lm_labels=None  # [src_len,batch_size]
                ):
        _, all_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = all_encoder_outputs[-1]  # 取Bert最后一层的输出
        # sequence_output: [src_len, batch_size, hidden_size]
        prediction_scores = self.classifier(sequence_output)
        # prediction_scores: [src_len, batch_size, vocab_size]
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.config.vocab_size),
                                      masked_lm_labels.reshape(-1))
            return prediction_scores,masked_lm_loss
        else:
            return prediction_scores  # [src_len, batch_size, vocab_size]


class BertForMaskedLM_pre(nn.Module):
    """
    仅为掩码语言预测模型
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForMaskedLM_pre, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        weights = None
        if config.use_embedding_weight:
            weights = self.bert.bert_embeddings.word_embeddings.embedding.weight
            logging.info(f"## 使用token embedding中的权重矩阵作为输出层的权重！{weights.shape}")
        self.classifier = BertForLMTransformHead(config, weights)
        self.config = config

    @classmethod
    def from_pretrained(cls, config, pretrained_model_dir=None, model_path=None):
        """
        从预训练模型目录或文件加载模型

        Args:
            config: BertConfig 配置对象
            pretrained_model_dir: 预训练模型目录路径（包含 pytorch_model.bin）
            model_path: 直接指定模型文件路径（如 model_ms_225.bin）

        Returns:
            加载了预训练权重的 BertForMaskedLM_pre 模型实例
        """
        import torch
        import os

        # 创建模型实例
        model = cls(config, bert_pretrained_model_dir=None)

        # 如果提供了直接的模型文件路径
        if model_path is not None and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

            # 记录加载信息
            if missing_keys:
                logging.warning(f"加载模型时缺少以下键 ({len(missing_keys)}个): {missing_keys[:5]}")
            if unexpected_keys:
                logging.warning(f"加载模型时发现未预期的键 ({len(unexpected_keys)}个): {unexpected_keys[:5]}")
            logging.info(f"成功从 {model_path} 加载模型权重 (strict=False)")
        # 如果提供了模型目录
        elif pretrained_model_dir is not None:
            # 尝试标准的 pytorch_model.bin
            standard_path = os.path.join(pretrained_model_dir, "pytorch_model.bin")
            if os.path.exists(standard_path):
                checkpoint = torch.load(standard_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

                # 记录加载信息
                if missing_keys:
                    logging.warning(f"加载模型时缺少以下键 ({len(missing_keys)}个): {missing_keys[:5]}")
                if unexpected_keys:
                    logging.warning(f"加载模型时发现未预期的键 ({len(unexpected_keys)}个): {unexpected_keys[:5]}")
                logging.info(f"成功从 {standard_path} 加载模型权重 (strict=False)")
            else:
                # 尝试使用 BertModel 的 from_pretrained（用于加载 BERT 基础权重）
                model.bert = BertModel.from_pretrained(config, pretrained_model_dir)
                logging.info(f"成功从 {pretrained_model_dir} 加载 BERT 基础权重")

        return model

    def forward(self,
                input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size]
                position_ids=None,
                masked_lm_labels=None  # [src_len,batch_size]
                ):
        _, all_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = all_encoder_outputs[-1]  # 取Bert最后一层的输出
        # sequence_output: [src_len, batch_size, hidden_size]
        prediction_scores = self.classifier(sequence_output)
        # prediction_scores: [src_len, batch_size, vocab_size]
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.config.vocab_size),
                                      masked_lm_labels.reshape(-1))
            return prediction_scores,masked_lm_loss,sequence_output
        else:
            return prediction_scores  # [src_len, batch_size, vocab_size]

class BertForPretrainingModel(nn.Module):
    """
    BERT预训练模型，包括MLM和NSP两个任务
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForPretrainingModel, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:  # 如果没有指定预训练模型路径，则随机初始化整个网络权重
            self.bert = BertModel(config)
        weights = None
        if 'use_embedding_weight' in config.__dict__ and config.use_embedding_weight:
            weights = self.bert.bert_embeddings.word_embeddings.embedding.weight
            logging.info(f"## 使用token embedding中的权重矩阵作为输出层的权重！{weights.shape}")
        self.mlm_prediction = BertForLMTransformHead(config, weights)
        self.nsp_prediction = nn.Linear(config.hidden_size, 2)
        self.config = config

    def forward(self, input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size]
                position_ids=None,
                masked_lm_labels=None,  # [src_len,batch_size]
                next_sentence_labels=None):  # [batch_size]
        pooled_output, all_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = all_encoder_outputs[-1]  # 取Bert最后一层的输出
        # sequence_output: [src_len, batch_size, hidden_size]
        mlm_prediction_logits = self.mlm_prediction(sequence_output)
        # mlm_prediction_logits: [src_len, batch_size, vocab_size]
        nsp_pred_logits = self.nsp_prediction(pooled_output)
        # # nsp_pred_logits： [batch_size, 2]
        if next_sentence_labels is not None:
            loss_fct_mlm = nn.CrossEntropyLoss(ignore_index=0)
            # MLM任务在构造数据集时pandding部分和MASK部分都是用的0来填充，所以ignore_index需要指定为0
            loss_fct_nsp = nn.CrossEntropyLoss()
            # 由于NSP中的分类标签中含有0，上面MLM中的损失指定了ignore_index=0，所以这里需要重新定义一个CrossEntropyLoss
            # 如果MLM任务在padding和MASK中用100之类的来代替，那么两者可以共用一个CrossEntropyLoss
            mlm_loss = loss_fct_mlm(mlm_prediction_logits.reshape(-1, self.config.vocab_size),
                                    masked_lm_labels.reshape(-1))
            nsp_loss = loss_fct_nsp(nsp_pred_logits.reshape(-1, 2),
                                   next_sentence_labels.reshape(-1))
            total_loss = mlm_loss
            # total_loss = mlm_loss + nsp_loss
            return total_loss, mlm_prediction_logits
        # else:
        #     return mlm_prediction_logits
        elif masked_lm_labels is not None:
            loss_fct_mlm = nn.CrossEntropyLoss(ignore_index=0)
            # MLM任务在构造数据集时pandding部分和MASK部分都是用的0来填充，所以ignore_index需要指定为0
            # 由于NSP中的分类标签中含有0，上面MLM中的损失指定了ignore_index=0，所以这里需要重新定义一个CrossEntropyLoss
            # 如果MLM任务在padding和MASK中用100之类的来代替，那么两者可以共用一个CrossEntropyLoss
            mlm_loss = loss_fct_mlm(mlm_prediction_logits.reshape(-1, self.config.vocab_size),
                                    masked_lm_labels.reshape(-1))
            total_loss = mlm_loss
            # total_loss = mlm_loss + nsp_loss
            return total_loss, mlm_prediction_logits
        else:
            return mlm_prediction_logits
        # [src_len, batch_size, vocab_size], [batch_size, 2]
