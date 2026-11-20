import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.BasicBert.BertEmbedding import BertEmbeddings


class BertEmbeddingsAblation(BertEmbeddings):
    """
    消融实验用的 embedding 层。
    通过 use_position / use_segment 两个开关控制哪些组件参与求和，
    其余逻辑与原 BertEmbeddings 完全一致。
    """

    def __init__(self, config, use_position: bool = True, use_segment: bool = True):
        super().__init__(config)
        self.use_position = use_position
        self.use_segment = use_segment

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None):
        """
        :param input_ids:       [src_len, batch_size]
        :param position_ids:    [1, src_len]
        :param token_type_ids:  [src_len, batch_size]
        :return:                [src_len, batch_size, hidden_size]
        """
        src_len = input_ids.size(0)

        embeddings = self.word_embeddings(input_ids)
        # [src_len, batch_size, hidden_size]

        if self.use_position:
            if position_ids is None:
                position_ids = self.position_ids[:, :src_len]
            embeddings = embeddings + self.position_embeddings(position_ids)

        if self.use_segment:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids, device=self.position_ids.device)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
