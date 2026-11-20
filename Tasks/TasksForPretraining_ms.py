import os
import numpy as np
import sys
from copy import deepcopy
sys.path.append('../')
from matchms_ws.importing import load_from_msp
import matchms_ws.filtering as ms_filters
from model import BertForPretrainingModel,BertForMaskedLM
from transformers import BertTokenizer
from model import BertConfig
from utils import LoadBertPretrainingDatasetMS
from transformers import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from utils import logger_init
# from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report
import logging
import torch
import time
import logging
import random
from tqdm import tqdm
# from .data_helpers import build_vocab
# from .data_helpers import pad_sequence
from torch.utils.data import DataLoader


# spectrums = list(load_from_msp("../data/test.msp"))
# # len(spectrums)
# spectrums=spectrums[:100]

# def metadata_processing(spectrum):
#     spectrum = ms_filters.default_filters(spectrum)
#     spectrum = ms_filters.repair_inchi_inchikey_smiles(spectrum)
#     spectrum = ms_filters.derive_inchi_from_smiles(spectrum)
#     spectrum = ms_filters.derive_smiles_from_inchi(spectrum)
#     spectrum = ms_filters.derive_inchikey_from_inchi(spectrum)
#     spectrum = ms_filters.harmonize_undefined_smiles(spectrum)
#     spectrum = ms_filters.harmonize_undefined_inchi(spectrum)
#     spectrum = ms_filters.harmonize_undefined_inchikey(spectrum)
#     spectrum = ms_filters.add_precursor_mz(spectrum)
#     return spectrum
# def peak_processing(spectrum):
#     spectrum = ms_filters.default_filters(spectrum)
#     spectrum = ms_filters.normalize_intensities(spectrum)
#     spectrum = ms_filters.select_by_intensity(spectrum, intensity_from=0.01)
#     spectrum = ms_filters.select_by_mz(spectrum, mz_from=10, mz_to=1000)
#     return spectrum

# spectrums = [metadata_processing(s) for s in spectrums]
# spectrums = [peak_processing(s) for s in spectrums]

# #把过滤后没有峰的化合物删掉
# new_spectrums=[]
# for i in range(len(spectrums)):
#     if len(spectrums[i].mz)>=1 and spectrums[i].peaks.intensities.max() == 1:
#         new_spectrums.append(spectrums[i])


# spectrum_documents = [SpectrumDocument(s, n_decimals=2) for s in new_spectrums]

# aa=[]
# for i in range(len(spectrum_documents)):
#     aa.append(" ".join(spectrum_documents[i].words))
from datetime import datetime


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ========== wike2 数据集相关配置
        # self.dataset_dir = os.path.join(self.project_dir, 'data', 'WikiText')
        # self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        # self.train_file_path = os.path.join(self.dataset_dir, 'wiki.train.tokens')
        # self.val_file_path = os.path.join(self.dataset_dir, 'wiki.valid.tokens')
        # self.test_file_path = os.path.join(self.dataset_dir, 'wiki.test.tokens')
        # self.data_name = 'wiki2'

        # ========== songci 数据集相关配置
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'ms')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_ms")
        self.train_file_path = os.path.join(self.dataset_dir, 'gnps_all_cleand_by_matchms_and_pubchem_lookups_62081_s2v_qian_no1000.msp')
        self.val_file_path = os.path.join(self.dataset_dir, 'MoNA-export-LC-MS-MS_Positive_Mode_val.msp')
        self.test_file_path = os.path.join(self.dataset_dir, 'MoNA-export-LC-MS-MS_Positive_Mode_test.msp')
        # self.train_file_path = os.path.join(self.dataset_dir, 'test.msp')
        # self.val_file_path = os.path.join(self.dataset_dir, 'test.msp')
        # self.test_file_path = os.path.join(self.dataset_dir, 'test.msp')
        self.data_name = 'ms'

        # 如果需要切换数据集，只需要更改上面的配置即可
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = os.path.join(self.pretrained_model_dir, f'cache_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        # self.model_save_dir = os.path.join(self.pretrained_model_dir, 'cache_20240325_181006')
        self.logs_save_dir = os.path.join(self.pretrained_model_dir, 'logs')
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{self.data_name}.bin')
        # self.model_save_path = os.path.join(self.model_save_dir, 'model_ms_5.bin')
        #self.writer = SummaryWriter(f"runs/{self.data_name}")
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 64
        self.max_sen_len = None  # 为None时则采用每个batch中最长的样本对该batch中的样本进行padding
        self.pad_index = 0
        self.random_state = 12022
        self.learning_rate = 1e-5
        self.weight_decay = 0.1
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.log_level = logging.DEBUG
        self.use_torch_multi_head = True  # False表示使用model/BasicBert/MyTransformer中的多头实现
        self.epochs = 150000
        self.model_val_per_epoch = 1
        self.n_decimals=2

        logger_init(log_file_name=self.data_name, log_level=self.log_level,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")
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


def accuracy(mlm_logits, mlm_labels, PAD_IDX):
    """
    :param mlm_logits:  [src_len,batch_size,src_vocab_size]
    :param mlm_labels:  [src_len,batch_size]
    :param nsp_logits:  [batch_size,2]
    :param nsp_label:  [batch_size]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [src_len,batch_size,src_vocab_size] 转成 [batch_size, src_len,src_vocab_size]
    mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
    # 将 [src_len,batch_size] 转成 [batch_size， src_len]
    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况，得到预测正确的个数（此时还包括有mask位置）
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉mlm_acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total

    # nsp_correct = (nsp_logits.argmax(1) == nsp_label).float().sum()
    # nsp_total = len(nsp_label)
    # nsp_acc = float(nsp_correct) / nsp_total
    return [mlm_acc, mlm_correct, mlm_total]

def evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    mlm_corrects, mlm_totals, nsp_corrects, nsp_totals = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (b_token_ids, b_mask, b_mlm_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
           
            mlm_logits,loss = model(input_ids=b_token_ids,
                                           attention_mask=b_mask,
                                           masked_lm_labels=b_mlm_label,
                                           token_type_ids=None)
            result = accuracy(mlm_logits, b_mlm_label, PAD_IDX)
            _, mlm_cor, mlm_tot = result
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
    model.train()
    return [float(mlm_corrects) / mlm_totals,loss]

def train(config):
    
    
    model = BertForPretrainingModel(config,
                                    None)
    last_epoch = -1
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        last_epoch = checkpoint['last_epoch']
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDatasetMS(vocab_path=config.vocab_path,
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
                                             n_decimals=config.n_decimals)
    train_iter, test_iter, val_iter = \
        data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                             train_file_path=config.train_file_path,
                                             val_file_path=config.val_file_path)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            "initial_lr": config.learning_rate

        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "initial_lr": config.learning_rate
        },
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                            int(len(train_iter) * 0),
                                                            int(config.epochs * len(train_iter)),
                                                            last_epoch=last_epoch)
    max_acc = 0
    state_dict = None
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_mask, b_mlm_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            
            loss,mlm_logits = model(input_ids=b_token_ids,
                                                    attention_mask=b_mask,
                                                    token_type_ids=None,
                                                    masked_lm_labels=b_mlm_label,
                                                    next_sentence_labels=None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses += loss.item()
            mlm_acc, _, _= accuracy(mlm_logits, b_mlm_label,
                                                    data_loader.PAD_IDX)
            if idx % 20 == 0:
                logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Batch[{idx}/{len(train_iter)}], "
                                f"Train loss :{loss.item():.3f}, Train mlm acc: {mlm_acc:.3f},")
                # config.writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                # config.writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
                # config.writer.add_scalars(main_tag='Training/Accuracy',
                #                           tag_scalar_dict={'NSP': nsp_acc,
                #                                            'MLM': mlm_acc},
                #                           global_step=scheduler.last_epoch)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Train loss: "
                        f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            mlm_acc = evaluate(config, val_iter, model, data_loader.PAD_IDX)[0]
            logging.info(f" ### MLM Accuracy on val: {round(mlm_acc, 4)}, ")
            # config.writer.add_scalars(main_tag='Testing/Accuracy',
            #                           tag_scalar_dict={'NSP': nsp_acc,
            #                                            'MLM': mlm_acc},
            #                           global_step=scheduler.last_epoch)
            # mlm_acc, nsp_acc = evaluate(config, train_iter, model, data_loader.PAD_IDX)
            if mlm_acc > max_acc:
                max_acc = mlm_acc
                state_dict = deepcopy(model.state_dict())
            torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                        os.path.join(config.model_save_dir, f'model_{config.data_name}_{epoch + 1}.bin'))
            logging.info(f"Save: [{epoch + 1}/{config.epochs}], model_{config.data_name}_{epoch + 1}.bin has been saved")
    torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                        os.path.join(config.model_save_dir, f'model_{config.data_name}.bin'))


def train1(config):
    # model = BertForPretrainingModel(config,
    #                                 config.pretrained_model_dir)
    model = BertForMaskedLM(config,
                                    None)
    last_epoch = -1
    print(config.model_save_path)
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        last_epoch = checkpoint['last_epoch']
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDatasetMS(vocab_path=config.vocab_path,
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
                                             n_decimals=config.n_decimals)
    train_iter, test_iter, val_iter = \
        data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                             train_file_path=config.train_file_path,
                                             val_file_path=config.val_file_path)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            "initial_lr": config.learning_rate

        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "initial_lr": config.learning_rate
        },
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=config.weight_decay)
    # optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          int(len(train_iter) * 0),
                                                          int(config.epochs * len(train_iter)),
                                                          last_epoch=last_epoch)
    max_acc = 0
    state_dict = None
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids,  b_mask, b_mlm_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
           
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            
            mlm_logits,masked_lm_loss = model(input_ids=b_token_ids,
                                                 attention_mask=b_mask,
                                                 token_type_ids=None,
                                                 masked_lm_labels=b_mlm_label)
            optimizer.zero_grad()
            masked_lm_loss.backward()
            optimizer.step()
            scheduler.step()
            losses += masked_lm_loss.item()
            mlm_acc, _, _, = accuracy(mlm_logits, b_mlm_label,
                                                    data_loader.PAD_IDX)
            if idx % 20 == 0:
                logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{masked_lm_loss.item():.3f}, Train mlm acc: {mlm_acc:.3f},"
                             )
                # config.writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                # config.writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
                # config.writer.add_scalars(main_tag='Training/Accuracy',
                #                           tag_scalar_dict={'NSP': nsp_acc,
                #                                            'MLM': mlm_acc},
                #                           global_step=scheduler.last_epoch)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Train loss: "
                     f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            mlm_acc,loss = evaluate(config, val_iter, model, data_loader.PAD_IDX)
            logging.info(f" ### MLM Accuracy on val: {round(mlm_acc, 4)}, Loss on val: {loss:.3f} "
                         )
            # config.writer.add_scalars(main_tag='Testing/Accuracy',
            #                           tag_scalar_dict={'NSP': nsp_acc,
            #                                            'MLM': mlm_acc},
            #                           global_step=scheduler.last_epoch)
            # mlm_acc, nsp_acc = evaluate(config, train_iter, model, data_loader.PAD_IDX)
            if mlm_acc > max_acc:
                max_acc = mlm_acc
                state_dict = deepcopy(model.state_dict())
        if (epoch + 1) % 50 == 0:
            torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                        os.path.join(config.model_save_dir, f'model_{config.data_name}_{epoch + 1}.bin'))
            logging.info(f"Save: [{epoch + 1}/{config.epochs}], model_{config.data_name}_{epoch + 1}.bin has been saved")
    torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                        os.path.join(config.model_save_dir, f'model_{config.data_name}.bin'))



# masked_rate=0.15
# masked_token_rate=0.8
# masked_token_unchanged_rate=0.5
# vocab = build_vocab("../bert_base_ms/vocab.txt")
# SEP_IDX = vocab['[SEP]']
# CLS_IDX = vocab['[CLS]']
# MASK_IDS = vocab['[MASK]']
# PAD_IDX = 0
if __name__ == '__main__':
    config = ModelConfig()
    train1(config)
    # sentences_1 = ["I no longer love her, true, but perhaps I love her.",
    #                "Love is so short and oblivion so long."]
    # sentences_2 = ["十年生死两茫茫。不思量。自难忘。千里孤坟，无处话凄凉。",
    #                "红酥手。黄藤酒。满园春色宫墙柳。"]
    # inference(config, sentences_2, masked=False, language='zh',random_state=2022)