import random
from argparse import ArgumentParser

from modeling_distillemb import BertModel, BertForSequenceClassification, BertForEmbeddingLM
from distill_emb import DistillEmbSmall, DistillEmb
from config import DistillModelConfig, DistillEmbConfig
import torch
import os
import re
import numpy as np
import lightning as L
import torch
import torch.nn as nn
# from lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from lightning.pytorch.loggers import WandbLogger
from distill_emb import DistillEmb
from tokenizer import CharTokenizer as Tokenizer
import json
from torch.utils.data import Dataset 
from config import DistillEmbConfig
from knn_classifier import KNNTextClassifier
from data_loader import load_sentiment
from data_loader import load_news_dataset
import pandas as pd
from retrieval import build_json_pairs, top1_accuracy
from torch.nn import functional as F
from loss_fns import generate_similars_from_embeddings, info_nce_loss
from tokenizer import CharTokenizer

random.seed(1000)
torch.random.manual_seed(10000)
np.random.seed(1000)

def load_word_embeddings(file_path: str, target_words: set = None, header: bool = True, word_prob=1.0) -> dict:
    word2vec = {}
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        if header:
            line = f.readline()
            n_vecs, dim = int(line.split(' ')[0]), int(line.split(' ')[1])
        for line in f:
            if random.random() < word_prob:
                line = line.strip().split(' ')
                word = line[0]
                vec = line[1:]
                if (target_words == None or word in target_words) and len(vec) == dim:
                    word2vec[word] = np.array([float(x) for x in vec], dtype=np.float32)
    print(f"Vec size: {len(word2vec[word])}")
    return word2vec

# dataset 
class DistillEmbDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: Tokenizer, wvectors: dict, vec_model=None, vec_size=512, max_length=512):
        self.data = data
        # set the column names to 'lang' and 'sentence'
        self.data.columns = ['lang', 'sentence']
        self.tokenizer = tokenizer
        self.wvectors = wvectors
        self.vec_size = vec_size
        self.vec_model = vec_model
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def get_word_vector(self, word: str):
        if word in self.wvectors:
            return self.wvectors[word]
        else:
            o = self.tokenizer.encode(word, return_attention_mask=False, add_cls=False, add_sep=False, return_tensors="pt")
            with torch.no_grad():
                vec = self.vec_model(**o)[0][0]
            return vec.cpu().numpy().tolist()

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        lang, sentence = row['lang'], row['sentence']
        sentence = self.tokenizer.clean_sentence(sentence)
        words = sentence.split()
        words = words[:self.max_length]  # limit to max word length
        sentence = ' '.join(words)
        # is_w2v = [True if w in self.wvectors else False for w in words]
        vecs = [self.get_word_vector(w) for w in words]
        chars = self.tokenizer.encode(sentence, return_attention_mask=False, add_cls=False, add_sep=False)['input_ids'][0]

        return {
            'lang': lang,
            'input_ids': chars,
            'input_embs': vecs,
        }

def collate_fn(batch, tokenizer: Tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    input_embs = [item['input_embs'] for item in batch]
    # is_w2v = [item['is_w2v'] for item in batch]
    attention_mask = [[1] * len(ids) for ids in input_ids]
    max_length = max([len(x) for x in input_ids])

    input_id_pad = tokenizer.encode(tokenizer.special_token2word['[PAD]'], add_cls=False, add_sep=False, return_attention_mask=False)['input_ids'][0][0]
    # print(torch.tensor([input_id_pad]).shape, torch.tensor(input_ids[0]).shape)
    input_ids = [np.array(ids + [input_id_pad] * (max_length - len(ids))) for ids in input_ids]
    # print([ids.shape for ids in input_ids])
    input_emb_pad = np.zeros((len(input_embs[0][0]), ), dtype=np.float32)

    input_embs = [np.array(emb + [input_emb_pad] * (max_length - len(emb))) for emb in input_embs]
    # is_w2v = [np.array(w2v + [True] * (max_length - len(w2v))) for w2v in is_w2v]
    attention_mask = [np.array(mask + [0] * (max_length - len(mask))) for mask in attention_mask]
    
    input_ids = np.array(input_ids)
    input_embs = np.array(input_embs)
    # is_w2v = np.array(is_w2v)
    attention_mask = np.array(attention_mask)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_embs = torch.tensor(input_embs, dtype=torch.float32)
    # is_w2v = torch.tensor(is_w2v, dtype=torch.int32)
    attention_mask = torch.tensor(attention_mask, dtype=torch.int32)


    assert input_ids.shape[0] == input_embs.shape[0]  == attention_mask.shape[0], f"Batch size mismatch: {input_ids.shape}, {input_embs.shape}, {attention_mask.shape}"
    assert input_ids.shape[1] == input_embs.shape[1]  == attention_mask.shape[1], f"Sequence length mismatch: {input_ids.shape}, {input_embs.shape},  {attention_mask.shape}"
    
    
    return {
        'input_ids': input_ids,
        'labels': input_embs,
        "attention_mask": attention_mask,
    }

tokenizer = CharTokenizer(charset_file_path='tokenizer/charset.json',
                          max_word_length=12)
config = DistillModelConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=4,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=2,
    pad_token_id=0,
    position_embedding_type="absolute",
    use_cache=True,
    classifier_dropout=None,
    embedding_type="distill",  # 'distilemb', 'fasttext'
    encoder_type='bert', #'lstm'
    num_input_chars=12,  # number of characters in each token
    char_vocab_size=tokenizer.char_vocab_size,
    distil_config=DistillEmbConfig(
        num_input_chars=tokenizer.max_word_length,  # number of characters in each token
        char_vocab_size=tokenizer.char_vocab_size,
        size="small",
        distill_dropout=0.1,
        embedding_size=512,  # size of the embedding vector for each character
    )
)
model = BertForSequenceClassification(config)
model.bert.load_word_embeddings('/home/leo/project/distil-research/distilemb/logs/distill_emb_v0/distill_emb_v0-epoch=136-epoch_val_loss=0.27.ckpt')


distill_emb = DistillEmb(config.distill_config)
state_dict = torch.load('logs/distill_emb_v0/distill_emb_v0-epoch=136-epoch_val_loss=0.27.ckpt', map_location='cpu')['state_dict']
# remove 'model.' prefix from state_dict keys
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
# load state_dict into distill_emb
distill_emb.load_state_dict(state_dict, strict=True)

w2v_emb = load_word_embeddings('embeddings/afriberta/afriberta.vec', word_prob=1.0)
data = pd.read_csv('pretrained-data/afriberta_train_lang.txt', delimiter="\t")
# take 50% of the data
data = data.sample(frac=0.5, random_state=42)
# split data into train and validation sets
train_data = data.sample(frac=0.9, random_state=42)
val_data = data.drop(train_data.index)

# dataset for distill emb
train_dataset = DistillEmbDataset(
    data=train_data,
    tokenizer=tokenizer,
    wvectors=w2v_emb,
    vec_model=distill_emb,
    vec_size=512,
    max_length=512
)

val_dataset = DistillEmbDataset(
    data=val_data,
    tokenizer=tokenizer,
    wvectors=w2v_emb,
    vec_model=distill_emb,
    vec_size=512,
    max_length=512
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=lambda x: collate_fn(x, train_dataset.tokenizer),
    num_workers=8,
    pin_memory=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=lambda x: collate_fn(x, val_dataset.tokenizer),
    num_workers=8,
    pin_memory=True,
    drop_last=False
)

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        embs = self.model(**kwargs).embeddings
        return embs


class BertEmbeddingLMModule(L.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.model = BertForEmbeddingLM(config)
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)
        self._extrinsic_eval(batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return [optimizer], [scheduler]
    
    def _extrinsic_eval(self, batch_idx):
        train_mode = self.model.training
        self.model.eval()  # Set the model to evaluation mode
        if batch_idx == 0:
            model = Wrapper(self.model)
            tokenizer = self.tokenizer
            classifier = KNNTextClassifier(tokenizer, model=model)

            df, classes = load_sentiment()
            # Sample equal amount for each language in the 'lang' column
            min_count = min(df['lang'].value_counts().min(), 250)
            sent_df = df.groupby('lang').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
            sent_train_df = sent_df.sample(frac=0.8, random_state=42)
            sent_test_df = sent_df.drop(sent_train_df.index)
            print(f"train shape: {sent_train_df.shape}, test shape: {sent_test_df.shape}")
            sent_f1, sent_acc, sent_per_lang, sent_test_df = classifier.classifiy(train_df=sent_train_df, test_df=sent_test_df, k=5, batch_size=8, model=None, tokenizer=None)

            df, classes = load_news_dataset()
            min_count = min(df['lang'].value_counts().min(), 250)
            news_df = df.groupby('lang').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
            news_train_df = news_df.sample(frac=0.8, random_state=42)
            news_test_df = news_df.drop(news_train_df.index)
            print(f"train shape: {news_train_df.shape}, test shape: {news_test_df.shape}")
            news_f1, news_acc, news_per_lang, news_test_df = classifier.classifiy(train_df=news_train_df, test_df=news_test_df, k=5, batch_size=8, model=None, tokenizer=None)

            df = pd.read_json('downstream-data/news_result.json')
            d = df.to_dict(orient='records')
            ret_acc, _, ret_per_lang = top1_accuracy(d, batch_size=8, model=model, tokenizer=tokenizer)

            self.log("sent_f1",sent_f1)
            self.log("news_f1", news_f1)
            self.log("retrieval_acc", ret_acc)
            self.log("task-average-f1", (sent_f1 + news_f1 + ret_acc) / 3.0)
        self.model.train(train_mode)  # Set the model back to training mode if it was in training mode before
    

embedding_lm_module = BertEmbeddingLMModule(config, tokenizer)


trainer = L.Trainer(
    max_epochs=1,
    logger=TensorBoardLogger("logs/", name="embedding_lm"),
    callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)],
    accelerator="auto",
    devices=1,
    log_every_n_steps=100,
    gradient_clip_algorithm="norm",
    gradient_clip_val=1.0,
    val_check_interval=10000,
    num_sanity_val_steps=0
)

trainer.fit(embedding_lm_module, train_loader, val_loader)
# Save the model
trainer.save_checkpoint("embedding_lm.ckpt")