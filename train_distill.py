import random
from argparse import ArgumentParser

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
from distill_emb import DistillEmbSmall as DistillEmb
from tokenizer import CharTokenizer as Tokenizer
import json
from torch.utils.data import Dataset 
from config import DistilEmbConfig
from transformers import AutoTokenizer, RwkvConfig, RwkvModel, AutoModel
from tokenizer import CharTokenizer
from knn_classifier import KNNTextClassifier
from data_loader import load_sentiment
from data_loader import load_news_dataset
import pandas as pd
from retrieval import build_json_pairs, top1_accuracy
from torch.nn import functional as F

random.seed(1000)
torch.random.manual_seed(10000)
np.random.seed(1000)

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def info_nce_loss_v2(anchor, positive, negatives, temperature=0.1):
    # Compute similarities
    sim_positive = F.cosine_similarity(anchor, positive, dim=-1)  # Similarity with positive
    sim_negatives = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1)  # Similarities with negatives
    
    # Scale by temperature
    sim_positive = sim_positive / temperature
    sim_negatives = sim_negatives / temperature
    
    # Exponentiate similarities
    exp_positive = torch.exp(sim_positive)
    exp_negatives = torch.exp(sim_negatives)
    
    # Compute denominator
    denominator = exp_positive + torch.sum(exp_negatives, dim=-1)
    
    # Compute InfoNCE loss
    loss = -torch.log(exp_positive / denominator)
    return loss.mean()  # Mean over batch

def info_nce_loss(q, k, queue=None, *, tau: float = 0.07, top_k: int | None = None):
    """
    InfoNCE with either
      • a per-sample negative queue  (queue shape  B × N × D), or
      • mined negatives from the mini-batch (queue=None).

    Args
    ----
    q      : (B, D)   – query embeddings
    k      : (B, D)   – positive key embeddings (same order as q)
    queue  : (B, N, D) or None
    tau    : float    – temperature
    top_k  : int or None
             • When queue is None: number of hardest negatives to keep per anchor.
               • None  -> use all B-1 batch negatives (SimCLR-style).
               • 1..B-2 -> keep that many hardest negatives.
             • Ignored if queue is provided.
    """
    # ℓ₂-normalise so dot-product ≈ cosine similarity
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    if queue is None:                               # -------- mined negatives --------
        B = q.size(0)

        # Full similarity matrix (B × B)
        sim = torch.matmul(q, k.t())                # sim_ij = <q_i , k_j>

        # Positive logits (diagonal)
        pos = sim.diag().unsqueeze(1)               # (B, 1)

        # All negatives: remove diagonal
        neg = sim.masked_select(~torch.eye(B, dtype=torch.bool,
                                           device=q.device))        # (B*(B-1),)
        neg = neg.view(B, B - 1)                                    # (B, B-1)

        # Keep only top-k hardest negatives if requested
        if top_k is not None and 0 < top_k < B - 1:
            neg, _ = torch.topk(neg, k=top_k, dim=1, largest=True)  # (B, top_k)

        # Assemble logits and compute CE
        logits = torch.cat([pos, neg], dim=1) / tau                 # (B, 1+neg)
        labels = torch.zeros(B, dtype=torch.long, device=q.device)  # positives at idx 0
        loss = F.cross_entropy(logits, labels)

    else:                                           # -------- explicit queue ---------
        queue = F.normalize(queue, dim=-1)          # (B, N, D)

        # Positive logits
        pos = torch.sum(q * k, dim=-1, keepdim=True)                # (B, 1)

        # Negative logits from queue
        neg = torch.einsum('bd,bnd->bn', q, queue)                  # (B, N)

        logits = torch.cat([pos, neg], dim=1) / tau                 # (B, 1+N)
        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits, labels)

    return loss




class DistillModule(L.LightningModule):
    def __init__(self, model: DistillEmb, tokenizer: Tokenizer, **kwargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.training_step_outputs = []

    def training_step(self, batch, batch_idx):

        x, pos_w2v, neg_w2v = batch
        z = self.model(x).view(pos_w2v.shape)
        
        # loss = self.triplet_loss(z, pos_w2v, neg_w2v)
        if neg_w2v != None and len(neg_w2v.shape) == 2:
            loss = self.triplet_loss(z, pos_w2v, neg_w2v)
        else:
            loss = info_nce_loss_v2(z, pos_w2v, neg_w2v)
        self.training_step_outputs.append(loss)
        self.log("train_loss", loss)
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        outputs = self.training_step_outputs
        loss = sum(outputs) / len(outputs)
        
        self._extrinsic_eval(self.current_epoch)
        self.model.train()
        self.log("epoch_train_loss", loss)
        self.training_step_outputs = []

    def validation_step(self, batch, batch_idx):
        x, pos_w2v, neg_w2v = batch
        z = self.model(x)

        # loss = self.triplet_loss(z, pos_w2v, neg_w2v)
        if neg_w2v != None and len(neg_w2v.shape) == 2:
            loss = self.triplet_loss(z, pos_w2v, neg_w2v)
        else:
            loss = info_nce_loss_v2(z, pos_w2v, neg_w2v)
        self.log("epoch_val_loss", loss)


    def configure_optimizers(self):
        
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, 
                                           weight_decay=self.hparams.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer ,
                              T_max = self.hparams.total_iteration, # Maximum number of iterations.
                             eta_min = 1e-6) # Minimum learning rate.
        
        return [self.optimizer], [self.scheduler]

    def _extrinsic_eval(self, current_epoch):
        train_mode = self.model.training
        self.model.eval()  # Set the model to evaluation mode
        if current_epoch % 16 == 0:
            model = self.model
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

def load_corpus_words(path, vocab_set, line_prob=1.0, min_sent_length=8):
    all_words = {}
    vocab2lang = {}
    langword2count = {}
    langword2count['punc'] = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if random.random() < line_prob:
                line = line.strip()
                lang, text = line.split('\t')
                
                if lang not in all_words:
                    all_words[lang] = []
                    langword2count[lang] = 0
                    
                words = text.split()
                words = [word for word in words if word in vocab_set]
                all_words[lang] += words
                
                for word in words:
                    if word not in vocab2lang:
                            vocab2lang[word] = {}
                            
                    if bool(re.match(r'^[\d\W]+$', word)):
                        if 'punc' not in vocab2lang[word]:
                            vocab2lang[word]['punc'] = 0
                        vocab2lang[word]['punc'] += 1
                        langword2count['punc'] += 1
                    else:
                        if lang not in vocab2lang[word]:
                            vocab2lang[word][lang] = 0
                        vocab2lang[word][lang] += 1
                        langword2count[lang] += 1
                    
                    
                    # check if word is  number of punctuation
                    
    
    # select the language for each vocab with highest count
    for word in vocab2lang:
        langs = vocab2lang[word]
        lang = max(langs, key=langs.get)
        vocab2lang[word] = lang
    
    temp = sum(langword2count.values())
    langword2count = {k: v / temp for k, v in langword2count.items()}

    return all_words, vocab2lang, langword2count

class LangDistillDataset(Dataset):
    
    def __init__(self, sentence_words, int2vocab, w2v_vectors, tokenizer: Tokenizer, neg_seq_len=32, top_k_negatives=1):
        self.sentence_words = sentence_words
        self.int2vocab = int2vocab
        self.wvectors = w2v_vectors
        
        self.neg_seq_len = neg_seq_len
        self.top_k_negatives = top_k_negatives

        self.tokenizer = tokenizer
        self.langs = list(sentence_words.keys())
        self.lang_prob = {lang: len(sentence_words[lang]) for lang in self.langs}
        self.lang_prob = {lang: self.lang_prob[lang] / sum(self.lang_prob.values()) for lang in self.langs}
        self.lang_prob = [self.lang_prob[lang] for lang in self.langs]

    def __len__(self):
        return len(self.int2vocab)
    
    def __getitem__(self, idx):
        
        target_word, lang = self.int2vocab[idx]
        target_vec = self.wvectors[target_word] # word2vec embedding
        
        neg_word = target_word
        xneg_words = []
        while target_word == neg_word:
            if lang == 'punc' or random.random() < 0.2:
                lang = random.choice(self.langs)

            end = random.randint(self.neg_seq_len + 1, len(self.sentence_words[lang]) - 1)
            start = end - self.neg_seq_len
            sent = self.sentence_words[lang][start:end]
            # print(target_word, sent)
            neg_words = set([self.int2vocab[x][0] for x in sent]) # incase there are duplicate words
            neg_words = neg_words - set([target_word]) # incase the target word is in the negative word
            neg_words = list(neg_words)
            
            if len(neg_words) < self.top_k_negatives:
                continue
            
            vecs = np.stack([self.wvectors[wrd] for wrd in neg_words]) # collect w2v embedding of negatives

            xsims = np.linalg.norm(vecs - target_vec, axis=1) # L2 norm (similarity)

            sims = np.argsort(xsims) # index of the most similar
            # neg_word = neg_words[sims[0]] # index the word with the negative word with the highest similarity to the postive
            xneg_words = [neg_words[xs] for xs in sims[:self.top_k_negatives]] # index the top k negatives
            neg_word = xneg_words[0] # the first negative word is the most similar to the target word

        # print(target_word, neg_word, sorted(neg_words)[:10])
        target_chars = self.tokenizer([target_word], add_special_tokens=False, return_attention_mask=False, padding=False, return_tensors="pt")['input_ids']
        target_chars = torch.LongTensor(target_chars).squeeze()
        pos_w2v = torch.Tensor(self.wvectors[target_word])
        # neg_w2v = torch.Tensor(self.wvectors[neg_word]).squeeze()
        neg_w2v = torch.Tensor(np.stack([self.wvectors[nw] for nw in xneg_words])).squeeze()
        return target_chars, pos_w2v, neg_w2v

def main():
    
    config = json.loads(open('distill_emb_v0.json').read())
    hparam = {k: v for k, v in config['train'].items()}
    for k, v in config['model'].items():
        hparam[k] = v
    print("Loaded training config")
    tokenizer = Tokenizer('tokenizer/charset.json', 
                          max_word_length=12)
    print("Loaded the tokenizer")
    w2v_emb = load_word_embeddings('embeddings/afriberta/afriberta.vec', word_prob=hparam['vector_load_ratio']) # load about 50% of the vectors
    
    vocab = set(w2v_emb.keys())
    if '</s>' in vocab:
        vocab.remove('</s>')
    print("Finished loading vectors")
    
    sentence_words, vocab2lang, langword2count = load_corpus_words('pretrained-data/afriberta_train_lang.txt', 
                                       vocab_set=vocab, 
                                       line_prob=hparam['sentence_load_ratio'], 
                                       min_sent_length=hparam['min_sent_length'])
    
    # intersection of vocab and vocab2lang
    vocab = vocab & set(vocab2lang.keys())
    
    vocab_list = list(vocab)
    np.random.shuffle(vocab_list)
    train_size = int(len(vocab) * hparam['train_ratio'])
    train_vocab = vocab_list[:train_size]
    test_vocab = vocab_list[train_size:]
    
    print("Finished loading corpus", len(sentence_words), sentence_words.keys())
    
    
    train_index2vocab = {k: [v, vocab2lang[v]] for k, v in enumerate(sorted(train_vocab))}
    train_vocab2index = {v: k for k, v in enumerate(sorted(train_vocab))}
    test_index2vocab = {k: [v, vocab2lang[v]] for k, v in enumerate(sorted(test_vocab))}
    test_vocab2index = {v: k for k, v in enumerate(sorted(test_vocab))}
    
    
    train_words = {}
    test_words = {}
    for lang in sentence_words.keys():
        train_words[lang] = []
        test_words[lang] = []
        for word in sentence_words[lang]:
            if word in train_vocab2index:
                train_words[lang].append(train_vocab2index[word])
            elif word in test_vocab2index:
                test_words[lang].append(test_vocab2index[word])
                
    print("# of train words: ", len(train_words))
    print("# of test words: ", len(test_words))
    print("# of train vocab: ", len(train_vocab))
    print("# of test vocab: ", len(test_vocab))
    print(train_words.keys())
    print(test_words.keys())

    train_dataset = LangDistillDataset(sentence_words=train_words,
                                int2vocab=train_index2vocab,  
                                w2v_vectors=w2v_emb, 
                                tokenizer=tokenizer, 
                                neg_seq_len=hparam['neg_seq_len'], top_k_negatives=3)

    # a, p, n = train_dataset[random.randint(0, len(train_dataset) - 1)]
    test_dataset = LangDistillDataset(sentence_words=test_words,  
                                  int2vocab=test_index2vocab,
                                    w2v_vectors=w2v_emb, 
                                    tokenizer=tokenizer, 
                                    neg_seq_len=hparam['neg_seq_len'], top_k_negatives=3)

    total_iteration = hparam['max_epochs'] * len(train_dataset) // hparam['batch_size']
    hparam['total_iteration'] = total_iteration
    train_dataloader = DataLoader(
        train_dataset, 
        num_workers=4, 
        pin_memory=True,
        shuffle=True,  
        batch_size=hparam['batch_size'])
    test_dataloader = DataLoader(
        test_dataset, batch_size=hparam['batch_size'], num_workers=2, pin_memory=True, shuffle=False)

    config = DistilEmbConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=9,
        num_attention_heads=8,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        embedding_type="distill",  # 'distilemb', 'fasttext'
        encoder_type='lstm',
        num_input_chars=tokenizer.max_word_length,  # number of characters in each token
        char_vocab_size=tokenizer.char_vocab_size,
        output_emb_size=512
    )
    distill_emb = DistillEmb(config)
    

    cbs = []

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch_val_loss",
        mode="min",
        dirpath="logs/distill_emb_v0",
        filename="distill_emb_v0-{epoch:02d}-{epoch_val_loss:.2f}",
    )

    # logger = WandbLogger(log_model="all", save_dir="logs/", name="distill_emb_v0", project="distill_emb", entity="leobitz")
    logger = TensorBoardLogger("logs", name="distill_emb_v0")
    
    
    cbs.append(checkpoint_callback)
    trainer = L.Trainer(
                max_epochs=hparam['max_epochs'], 
                logger=logger, 
                log_every_n_steps=1,
                gradient_clip_val=hparam['clip_grad_norm'],
                gradient_clip_algorithm='norm', 
                callbacks=cbs)

    trainer.fit(model=DistillModule(model=distill_emb, tokenizer=tokenizer, **hparam),
                train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

if __name__ == '__main__':
    main()