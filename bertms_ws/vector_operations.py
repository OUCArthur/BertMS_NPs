import logging
from typing import Union
import numba
import numpy as np
from gensim.models.basemodel import BaseTopicModel
from .Document import Document
from matchms_ws import Spectrum
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from .Spec2Vec import SpectrumDocument
import matchms_ws.filtering as ms_filters

logger = logging.getLogger("bertms")


class Vocab:
    """
    根据本地的vocab文件，构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)

def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors

class LoadBertPretrainingDatasetMS(object):
    r"""

    Arguments:

    """

    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True,
                 random_state=22021,
                 data_name='wiki2',
                 masked_rate=0.15,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 seps="。",
                 n_decimals=2):
        self.tokenizer = tokenizer
        self.seps = seps
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.max_position_embeddings = max_position_embeddings
        self.pad_index = pad_index
        self.is_sample_shuffle = is_sample_shuffle
        self.data_name = data_name
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.random_state = random_state
        self.n_decimals=n_decimals
        random.seed(random_state)

    def get_format_data(self, filepath):
        """
        本函数的作用是将数据集格式化成标准形式
        :param filepath:
        :return:  [ [sentence 1, sentence 2, ...], [sentence 1, sentence 2,...],...,[] ]
        """
        if self.data_name == 'wiki2':
            return read_wiki2(filepath, self.seps)
        elif self.data_name == 'custom':
            return read_custom(filepath)
            # 在这里，可以调用你自己数据对应的格式化函数，
            # 但是返回格式需要同read_wiki2()保持一致。
        elif self.data_name == 'songci':
            return read_songci(filepath, self.seps)
        else:
            raise ValueError(f"数据 {self.data_name} 不存在对应的格式化函数，"
                             f"请参考函数 read_wiki(filepath) 实现对应的格式化函数！")

    @staticmethod
    def get_next_sentence_sample(sentence, next_sentence, paragraphs):
        """
        本函数的作用是根据给定的连续两句话和对应的段落，返回NSP任务中的句子对和标签
        :param sentence:  str
        :param next_sentence: str
        :param paragraphs: [str,str,...,]
        :return: sentence A, sentence B, True
        """
        if random.random() < 0.5:  # 产生[0,1)之间的一个随机数
            is_next = True
        else:
            # 这里random.choice的作用是从一个list中随机选出一个元素
            # ①先从所有段落中随机出一个段落；
            # ②再从随机出的一个段落中随机出一句话；
            new_next_sentence = next_sentence
            while next_sentence == new_next_sentence: # 防止随机选择的下一个句子仍旧与之前的相同（尽管概率非常小）
                new_next_sentence = random.choice(random.choice(paragraphs))
            next_sentence = new_next_sentence
            is_next = False
        return sentence, next_sentence, is_next

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_token_rate:  # 0.8
                masked_token_id = self.MASK_IDS
            else:
                # 10%的时间：保持词不变
                if random.random() < self.masked_token_unchanged_rate:  # 0.5
                    masked_token_id = token_ids[mlm_pred_position]
                # 10%的时间：用随机词替换该词
                else:
                    masked_token_id = random.randint(0, len(self.vocab.stoi) - 1)
            mlm_input_tokens_id[mlm_pred_position] = masked_token_id
            pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        # num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))
        num_mlm_preds = max(0, round(len(token_ids) * self.masked_rate))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label



    
    def data_process(self, filepath, postfix='cache'):
        """
        本函数的作用是是根据格式化后的数据制作NSP和MLM两个任务对应的处理完成的数据
        :param filepath:
        :return:
        """
        # paragraphs = read_songci("/home/wus/BertWithPretrained/data/SongCi/test.json", "。")
        # spectrums = list(load_from_msp(filepath))
        spectrums = filepath
        # spectrums = self.metadata_processing(spectrums)
        # spectrums = self.peak_processing(spectrums)
        # print(spectrums.peaks.mz)
        spectrum_documents=SpectrumDocument(spectrums, n_decimals=self.n_decimals)
        if len(spectrum_documents.words)<512:
            spectrum_documents_words=" ".join(spectrum_documents.words)
        else:
            spectrum_documents_words=" ".join(spectrum_documents.words[:512])
        paragraphs=spectrum_documents_words
        # 返回的是一个二维列表，每个列表可以看做是一个段落（其中每个元素为一句话）
        data = []
        max_len = 0
        # 这里的max_len用来记录整个数据集中最长序列的长度，在后续可将其作为padding长度的标准
        desc = f" ## 正在构造NSP和MLM样本({filepath})"
        # print(paragraphs)
         # 遍历每个
        sentence=paragraphs
        # for i in range(len(paragraph) - 1):  # 遍历一个段落中的每一句话
        #     sentence, next_sentence, is_next = get_next_sentence_sample(
        #         paragraph[i], paragraph[i + 1], paragraphs)  # 构造NSP样本
        #     print(f" ## 当前句文本：{sentence}")
        #     print(f" ## 下一句文本：{next_sentence}")
        #     print(f" ## 下一句标签：{is_next}")
            # if len(next_sentence) < 2:
            #     logging.warning(f"句子'{sentence}'的下一句为空，请检查数据预处理。 当前段落文本为{paragraph}")
            #     continue
        token_a_ids = [self.vocab[token] for token in sentence.split()]
        # token_b_ids = [vocab[token] for token in tokenizer(next_sentence)]
        # token_ids = [CLS_IDX] + token_a_ids + [SEP_IDX] + token_b_ids
        token_ids = [self.CLS_IDX] + token_a_ids + [self.SEP_IDX]
        # seg1 = [0] * (len(token_a_ids) + 2)  # 2 表示[CLS]和中间的[SEP]这两个字符
        # seg2 = [1] * (len(token_b_ids) + 1)
        # segs = seg1 + seg2
        # segs = seg1
        if len(token_ids) > 512 - 1:
            token_ids = token_ids[:512 - 1]  # BERT预训练模型只取前512个字符
            # segs = segs[:512]
        # token_ids += [SEP_IDX]
        assert len(token_ids) <= 512
        # assert len(segs) <= 512
        # print(f" ## Mask之前词元结果：{[self.vocab.itos[t] for t in token_ids]}")
        # segs = torch.tensor(segs, dtype=torch.long)
        # print(f" ## Mask之前token ids:{token_ids}")
        # print(f" ##      segment ids:{segs.tolist()},序列长度为 {len(segs)}")
        # nsp_lable = torch.tensor(int(is_next), dtype=torch.long)
        mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
        token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
        mlm_label = torch.tensor(mlm_label, dtype=torch.long)
        max_len = max(max_len, token_ids.size(0))
        # mlm_intensities=torch.tensor(spectrums.peaks.intensities, dtype=torch.long)
        # print(f" ## Mask之后token ids:{token_ids.tolist()}")
        # print(f" ## Mask之后词元结果：{[self.vocab.itos[t] for t in token_ids.tolist()]}")
        # print(f" ## Mask之后label ids:{mlm_label.tolist()}")
        # print(f" ## 当前样本构造结束================== \n\n")
        # data.append([token_ids, segs, nsp_lable, mlm_label])
        data.append([token_ids, mlm_label])
        all_data = {'data': data, 'max_len': max_len}
        return all_data

    def generate_batch(self, data_batch):
        b_token_ids, b_mlm_label = [], []
        for (token_ids, mlm_label) in data_batch:
            # 开始对一个batch中的每一个样本进行处理
            b_token_ids.append(token_ids)
            b_mlm_label.append(mlm_label)
        b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=114)
        # b_token_ids:  [src_len,batch_size]

        b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=114)
        # b_mlm_label:  [src_len,batch_size]

        b_mask = (b_token_ids == self.PAD_IDX).transpose(0, 1)
        # b_mask: [batch_size,max_len]

        return b_token_ids, b_mask, b_mlm_label

    def load_train_val_test_data(self,
                                 train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        postfix = f"_ml{self.max_sen_len}_rs{self.random_state}_mr{str(self.masked_rate)[2:]}" \
                  f"_mtr{str(self.masked_token_rate)[2:]}_mtur{str(self.masked_token_unchanged_rate)[2:]}"
        test_data = self.data_process(filepath=test_file_path,
                                      postfix='test' + postfix)['data']
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        # logging.info(f"## 成功返回测试集，一共包含样本{len(test_iter.dataset)}个")
        return test_iter

    def make_inference_samples(self, sentences=None, masked=False, language='en', random_state=None):
        """
        制作推理时的数据样本
        :param sentences:
        :param masked:  指传入的句子没有标记mask的位置
        :param language:  判断是中文zh还是英文en
        :param random_state:  控制mask字符时的随机状态
        :return:
        e.g.
        sentences = ["I no longer love her, true,but perhaps I love her.",
                     "Love is so short and oblivion so long."]
        input_tokens_ids.transpose(0,1):
                tensor([[  101,  1045,  2053,   103,  2293,  2014,  1010,  2995,  1010,  2021,
                            3383,   103,  2293,  2014,  1012,   102],
                        [  101,  2293,   103,  2061,  2460,  1998, 24034,  2061,  2146,  1012,
                            102,     0,     0,     0,     0,     0]])
        tokens:
                [CLS] i no [MASK] love her , true , but perhaps [MASK] love her . [SEP]
                [CLS] love [MASK] so short and oblivion so long . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD]
        pred_index:
                [[3, 11], [2]]
        mask:
                tensor([[False, False, False, False, False, False, False, False, False, False,
                        False, False, False, False, False, False],
                      [False, False, False, False, False, False, False, False, False, False,
                        False,  True,  True,  True,  True,  True]])
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
        mask_token = self.vocab.itos[self.MASK_IDS]
        input_tokens_ids = []
        pred_index = []
        for sen in sentences:
            if language == 'en':
                sen_list = sen.split()
            else:
                sen_list = [w for w in sen]
            tmp_token = []
            if not masked:  # 如果传入的样本没有进行mask，则此处进行mask
                candidate_pred_positions = [i for i in range(len(sen_list))]
                random.seed(random_state)
                random.shuffle(candidate_pred_positions)
                num_mlm_preds = max(1, round(len(sen_list) * self.masked_rate))
                for p in candidate_pred_positions[:num_mlm_preds]:
                    sen_list[p] = mask_token
            for item in sen_list:  # 逐个词进行tokenize
                if item == mask_token:
                    tmp_token.append(item)
                else:
                    tmp_token.extend(self.tokenizer(item))
            token_ids = [self.vocab[t] for t in tmp_token]
            token_ids = [self.CLS_IDX] + token_ids + [self.SEP_IDX]
            pred_index.append(self.get_pred_idx(token_ids))  # 得到被mask的Token的位置
            input_tokens_ids.append(torch.tensor(token_ids, dtype=torch.long))
        input_tokens_ids = pad_sequence(input_tokens_ids,
                                        padding_value=self.PAD_IDX,
                                        batch_first=False,
                                        max_len=None)  # 按一个batch中最长的样本进行padding
        mask = (input_tokens_ids == self.PAD_IDX).transpose(0, 1)
        return input_tokens_ids, pred_index, mask

    def get_pred_idx(self, token_ids):
        """
        根据token_ids返回'[MASK]'所在的位置，即需要预测的位置
        :param token_ids:
        :return:
        """
        pred_idx = []
        for i, t in enumerate(token_ids):
            if t == self.MASK_IDS:
                pred_idx.append(i)
        return pred_idx

def metadata_processing(spectrum):
    spectrum = ms_filters.default_filters(spectrum)
    spectrum = ms_filters.repair_inchi_inchikey_smiles(spectrum)
    spectrum = ms_filters.derive_inchi_from_smiles(spectrum)
    spectrum = ms_filters.derive_smiles_from_inchi(spectrum)
    spectrum = ms_filters.derive_inchikey_from_inchi(spectrum)
    spectrum = ms_filters.harmonize_undefined_smiles(spectrum)
    spectrum = ms_filters.harmonize_undefined_inchi(spectrum)
    spectrum = ms_filters.harmonize_undefined_inchikey(spectrum)
    spectrum = ms_filters.add_precursor_mz(spectrum)
    return spectrum
def peak_processing(spectrum):
    spectrum = ms_filters.default_filters(spectrum)
    spectrum = ms_filters.normalize_intensities(spectrum)
    spectrum = ms_filters.select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = ms_filters.select_by_mz(spectrum, mz_from=10, mz_to=1000)
    return spectrum


def calc_vector(model: BaseTopicModel, document: Document,
                intensity_weighting_power: Union[float, int] = 0,
                allowed_missing_percentage: Union[float, int] = 10) -> np.ndarray:
    """Compute document vector as a (weighted) sum of individual word vectors.

    Parameters
    ----------
    model
        Pretrained word2vec model to convert words into vectors.
    document
        Document containing document.words and document.weights.
    intensity_weighting_power
        Specify to what power weights should be raised. The default is 0, which
        means that no weighing will be done.
    allowed_missing_percentage:
        Set the maximum allowed percentage of the document that may be missing
        from the input model. This is measured as percentage of the weighted, missing
        words compared to all word vectors of the document. Default is 10, which
        means up to 10% missing words are allowed. If more words are missing from
        the model, an empty embedding will be returned (leading to similarities of 0)
        and a warning is raised.

    Returns
    -------
    vector
        Vector representing the input document in latent space. Will return None
        if the missing percentage of the document in the model is > allowed_missing_percentage.
    """
    assert max(document.weights) <= 1.0, "Weights are not normalized to unity as expected."
    assert 0 <= allowed_missing_percentage <= 100.0, "allowed_missing_percentage must be within [0,100]"

    def _check_model_coverage():
        """Return True if model covers enough of the document words."""
        if len(idx_not_in_model) > 0:
            weights_missing = np.array([document.weights[i] for i in idx_not_in_model])
            weights_missing_raised = np.power(weights_missing, intensity_weighting_power)
            missing_percentage = 100 * weights_missing_raised.sum() / (weights_raised.sum()
                                                                       + weights_missing_raised.sum())
            msg = (f"Found {len(idx_not_in_model)} word(s) missing in the model.",
                   f"Weighted missing percentage not covered by the given model is {missing_percentage:.2f}%.")
            logger.info(msg)

            if missing_percentage > allowed_missing_percentage:
                msg = (f"Missing percentage ({missing_percentage:.2f}%) is above set maximum. An empty vector will be returned.",
                       "Consider retraining the used model or change the `allowed_missing_percentage`.")
                logger.warning(msg)
                return False
        return True

    idx_not_in_model = [i for i, x in enumerate(document.words) if x not in model.wv.key_to_index]
    if len(idx_not_in_model) == len(document.words):
        msg = ("Spectrum without peaks known by the used model. An empty vector will be returned.",
               "Consider retraining the used model or make sure decimal rounding is correct.")
        logger.warning(msg)
        return np.zeros(model.wv.vector_size)

    words_in_model = [x for i, x in enumerate(document.words) if i not in idx_not_in_model]
    weights_in_model = np.asarray([x for i, x in enumerate(document.weights)
                                   if i not in idx_not_in_model]).reshape(len(words_in_model), 1)

    word_vectors = model.wv[words_in_model]
    weights_raised = np.power(weights_in_model, intensity_weighting_power)

    if _check_model_coverage() is True:
        weights_raised_tiled = np.tile(weights_raised, (1, model.wv.vector_size))
        return np.sum(word_vectors * weights_raised_tiled, 0)
    return np.zeros(model.wv.vector_size)


def calc_vector_bert(model: BaseTopicModel, document: Spectrum,
                intensity_weighting_power: Union[float, int] = 0,
                allowed_missing_percentage: Union[float, int] = 10) -> np.ndarray:
    data_loader = LoadBertPretrainingDatasetMS(vocab_path=model.config.vocab_path,
                                             tokenizer=None,
                                             batch_size=model.config.batch_size,
                                             max_sen_len=model.config.max_sen_len,
                                             max_position_embeddings=model.config.max_position_embeddings,
                                             pad_index=model.config.pad_index,
                                             is_sample_shuffle=model.config.is_sample_shuffle,
                                             random_state=model.config.random_state,
                                             data_name=model.config.data_name,
                                             masked_rate=model.config.masked_rate,
                                             masked_token_rate=model.config.masked_token_rate,
                                             masked_token_unchanged_rate=model.config.masked_token_unchanged_rate,
                                             n_decimals=model.config.n_decimals)
    # document = metadata_processing(document)
    # document = peak_processing(document)
    test_iter = \
        data_loader.load_train_val_test_data(test_file_path=document,
                                             train_file_path=None,
                                             val_file_path=None)
    sequence_output_sum = torch.zeros(len(test_iter.dataset),model.config.hidden_size)
    sequence_output_sum=sequence_output_sum.to(model.config.device)
    with torch.no_grad():
        for idx, (b_token_ids, b_mask, b_mlm_label) in enumerate(test_iter):
            b_token_ids = b_token_ids.to(model.config.device)  # [src_len, batch_size]
            b_mask = b_mask.to(model.config.device)
            b_mlm_label = b_mlm_label.to(model.config.device)
           
            _,_,sequence_output = model(input_ids=b_token_ids,
                                           attention_mask=b_mask,
                                           masked_lm_labels=b_mlm_label,
                                           token_type_ids=None)
            # print(sum,len(document.peaks.intensities),document.peaks.intensities)
            
            intensities=torch.tensor(document.peaks.intensities, dtype=torch.float64, device=model.config.device)
            sequence_output=sequence_output.transpose(1,0)
            for i in range(sequence_output.size(0)):
                min=113
                if intensities.size(0)<113:
                    min=intensities.size(0)
                for j in range(min):
                    if b_mask[i][j]==False:
                        sequence_output_sum[idx*model.config.batch_size+i]=sequence_output_sum[idx*model.config.batch_size+i]+torch.sqrt(intensities[j])*sequence_output[i][j+1]
                    else:
                        break
    model.train()
    return sequence_output_sum
    


def cosine_similarity_matrix(vectors_1, vectors_2):
    """Fast implementation of cosine similarity between two arrays of vectors.

    For example:

    .. code-block:: python

        import numpy as np
        from bertms_ws.vector_operations import cosine_similarity_matrix

        vectors_1 = np.array([[1, 1, 0, 0],
                              [1, 0, 1, 1]])
        vectors_2 = np.array([[0, 1, 1, 0],
                              [0, 0, 1, 1]])
        similarity_matrix = cosine_similarity_matrix(vectors_1, vectors_2)


    Parameters
    ----------
    vectors_1
        Numpy array of vectors. vectors_1.shape[0] is number of vectors, vectors_1.shape[1]
        is vector dimension.
    vectors_2
        Numpy array of vectors. vectors_2.shape[0] is number of vectors, vectors_2.shape[1]
        is vector dimension.
    """
    assert vectors_1.shape[1] == vectors_2.shape[1], "Input vectors must have same shape."
    # print(vectors_1,vectors_2)
    # print(vector1.dtype)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vectors_1=torch.tensor(vectors_1, device=device)
    # print(vector1.dtype)
    vectors_2=torch.tensor(vectors_2, device=device)
    cos_sim = torch.unsqueeze(F.cosine_similarity(vectors_1[0], vectors_2, dim=1), dim=0)
    # print(cos_sim)
    for i in range(1,vectors_1.size(0)):
        cos_sim1 = torch.unsqueeze(F.cosine_similarity(vectors_1[i], vectors_2, dim=1), dim=0)
        cos_sim=torch.cat((cos_sim,cos_sim1),dim=0)
    # print(cos_sim)
    return cos_sim
    # with torch.autograd.no_grad():
	# blabla

    # vectors_1=torch.as_tensor(vectors_1, dtype=torch.float64,device="cuda:0")
    # vectors_2=torch.as_tensor(vectors_2, dtype=torch.float64,device="cuda:0")
    # norm_1 = torch.sqrt(torch.sum(vectors_1**2, dim=1))
    # norm_2 = torch.sqrt(torch.sum(vectors_2**2, dim=1))
    # for i in range(vectors_1.shape[0]):
    #     vectors_1[i] = vectors_1[i] / norm_1[i]
    # for i in range(vectors_2.shape[0]):
    #     vectors_2[i] = vectors_2[i] / norm_2[i]
    # return torch.mm(vectors_1, vectors_2.T)



def cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two input vectors.

    For example:

    .. testcode::

        import numpy as np
        from bertms_ws.vector_operations import cosine_similarity

        vector1 = np.array([1, 1, 0, 0])
        vector2 = np.array([1, 1, 1, 1])
        print("Cosine similarity: {:.3f}".format(cosine_similarity(vector1, vector2)))

    Should output

    .. testoutput::

        Cosine similarity: 0.707

    Parameters
    ----------
    vector1
        Input vector. Can be array of integers or floats.
    vector2
        Input vector. Can be array of integers or floats.
    """
    assert vector1.shape[0] == vector2.shape[0], "Input vector must have same shape."
    # print(vector1.dtype)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vector1=torch.tensor(vector1, device=device)
    # print(vector1.dtype)
    vector2=torch.tensor(vector2, device=device)
    prod12 = 0
    prod11 = 0
    prod22 = 0
    for i in range(vector1.shape[0]):
        prod12 += vector1[i] * vector2[i]
        prod11 += vector1[i] * vector1[i]
        prod22 += vector2[i] * vector2[i]
    cosine_score=0.0
    if prod11 != 0 and prod22 != 0:
        cosine_score = prod12 / torch.sqrt(prod11 * prod22)
    print("cosine_score:"+str(cosine_score))
    return cosine_score
