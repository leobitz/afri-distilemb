import random
import re
import numpy as np
import lightning as L
import torch
import torch.nn as nn

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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn import functional as F
from helper import load_word_embeddings, load_corpus_words
from helper import LangDistillDataset
import argparse
import os
from huggingface_hub import HfApi, create_repo
import multiprocessing as mp

random.seed(1000)
torch.random.manual_seed(10000)
np.random.seed(1000)


class DistillModule(L.LightningModule):
    def __init__(self, model: DistillEmb, tokenizer: Tokenizer, **kwargs):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self.normalize = kwargs.get("normalize", True)
        self.temperature = kwargs.get("temperature", 0.1)
        
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.training_step_outputs = []
        self.best_task_performance = {}

    def training_step(self, batch, batch_idx):

        x, pos_w2v, neg_w2v = batch
        z = self.model(x).view(pos_w2v.shape)

        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)
            pos_w2v = F.normalize(pos_w2v, p=2, dim=-1)
            if neg_w2v != None and len(neg_w2v.shape) == 3:
                b, s, f = neg_w2v.shape
                neg_w2v = neg_w2v.view(b * s, f)
                neg_w2v = F.normalize(neg_w2v, p=2, dim=-1)
                neg_w2v = neg_w2v.view(b, s, f)
            elif neg_w2v != None and len(neg_w2v.shape) == 2:
                neg_w2v = F.normalize(neg_w2v, p=2, dim=-1)
        
        if neg_w2v != None and len(neg_w2v.shape) == 2:
            loss = self.triplet_loss(z, pos_w2v, neg_w2v)
        else:
            loss = info_nce_loss(z, pos_w2v, neg_w2v, temperature=self.temperature)
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

        z = self.model(x).view(pos_w2v.shape)
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)
            pos_w2v = F.normalize(pos_w2v, p=2, dim=-1)
            if neg_w2v != None and len(neg_w2v.shape) == 3:
                b, s, f = neg_w2v.shape
                neg_w2v = neg_w2v.view(b * s, f)
                neg_w2v = F.normalize(neg_w2v, p=2, dim=-1)
                neg_w2v = neg_w2v.view(b, s, f)
            elif neg_w2v != None and len(neg_w2v.shape) == 2:
                neg_w2v = F.normalize(neg_w2v, p=2, dim=-1)
        
        if neg_w2v != None and len(neg_w2v.shape) == 2:
            loss = self.triplet_loss(z, pos_w2v, neg_w2v)
        else:
            loss = info_nce_loss(z, pos_w2v, neg_w2v, temperature=self.temperature)
        self.log("epoch_val_loss", loss)

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.98),
        )
        warmup_iters = int(0.05 * self.hparams.total_iteration)
        cosine_iters = self.hparams.total_iteration - warmup_iters
        sched = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_iters)),
                CosineAnnealingLR(optimizer, T_max=max(1, cosine_iters), eta_min=1e-6),
            ],
            milestones=[warmup_iters],
        )
        return [optimizer], [{"scheduler": sched, "interval": "step"}]

    def _extrinsic_eval(self, current_epoch):
        train_mode = self.model.training
        self.model.eval()  # Set the model to evaluation mode
        if current_epoch % self.hparams.task_eval_every == 0:
            model = self.model
            tokenizer = self.tokenizer
            classifier = KNNTextClassifier(tokenizer, model=model)

            df, classes = load_sentiment()
            # Sample equal amount for each language in the 'lang' column
            min_count = min(df['lang'].value_counts().min(), 250)
            sent_df = df.groupby('lang').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
            sent_train_df = sent_df.sample(frac=0.8, random_state=42)
            sent_test_df = sent_df.drop(sent_train_df.index)
            sent_f1, sent_acc, sent_per_lang, sent_test_df = classifier.classifiy(train_df=sent_train_df, test_df=sent_test_df, k=5, batch_size=8, model=None, tokenizer=None)

            df, classes = load_news_dataset()
            min_count = min(df['lang'].value_counts().min(), 250)
            news_df = df.groupby('lang').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
            news_train_df = news_df.sample(frac=0.8, random_state=42)
            news_test_df = news_df.drop(news_train_df.index)
            news_f1, news_acc, news_per_lang, news_test_df = classifier.classifiy(train_df=news_train_df, test_df=news_test_df, k=5, batch_size=8, model=None, tokenizer=None)

            df = pd.read_json('downstream-data/news_result.json')
            d = df.to_dict(orient='records')
            ret_acc, _, ret_per_lang = top1_accuracy(d, batch_size=8, model=model, tokenizer=tokenizer)

            self.log("sent_f1",sent_f1)
            self.log("news_f1", news_f1)
            self.log("retrieval_acc", ret_acc)
            self.log("task-average-f1", (sent_f1 + news_f1 + ret_acc) / 3.0)
            
            avg_score = (sent_f1 + news_f1 + ret_acc) / 3.0
            current_best = self.best_task_performance.get("task-average-f1", float("-inf"))
            if avg_score > current_best:
                ckpt_path = f"logs/{self.hparams.run_name}/best_avg_epoch/"
                dir_name = ckpt_path #os.path.dirname(ckpt_path)
                self.best_task_performance = {
                    "sent_f1": sent_f1,
                    "news_f1": news_f1,
                    "retrieval_acc": ret_acc,
                    "task-average-f1": (sent_f1 + news_f1 + ret_acc) / 3.0,
                    "epoch": current_epoch,
                    "path": dir_name
                }
                os.makedirs(dir_name, exist_ok=True)
                self.model.save_pretrained(dir_name)
                with open(os.path.join(dir_name, "tokenizer_config.json"), "w") as f:
                    charset = json.load(open('tokenizer/charset.json', 'r', encoding='utf-8'))
                    json.dump(
                        {
                            "max_word_length": tokenizer.max_word_length,
                            "char_vocab_size": tokenizer.char_vocab_size,
                            "charset": charset
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                # save hyperparams
                with open(os.path.join(dir_name, "hparams.json"), "w") as f:
                    json.dump(self.hparams, f, ensure_ascii=False, indent=2)
                # save performance
                with open(os.path.join(dir_name, "performance.json"), "w") as f:
                    json.dump(self.best_task_performance, f, ensure_ascii=False, indent=2)
                print(f"New best average task performance at epoch {current_epoch}: {self.best_task_performance}")

        self.model.train(train_mode)  # Set the model back to training mode if it was in training mode before
    
def main(hparam: dict):
    print("Loaded training config")
    tokenizer = Tokenizer('tokenizer/charset.json', 
                          max_word_length=12)
    print("Loaded the tokenizer")
    w2v_emb = load_word_embeddings('embeddings/afriberta/afriberta.vec', word_prob=hparam['vector_load_ratio']) # load about 50% of the vectors
    
    vocab = set(w2v_emb.keys())
    if '</s>' in vocab:
        vocab.remove('</s>')
    print("Finished loading vectors")
    
    sentence_words, vocab2lang = load_corpus_words('pretrained-data/afriberta_train_lang.txt', 
                                       vocab_set=vocab, 
                                       line_prob=hparam['sentence_load_ratio'], 
                                       min_sent_length=hparam['min_sent_length'],
                                       lang2word_file='pretrained-data/lang2word.json',
                                       vocab2lang_file='pretrained-data/vocab2lang.json')
    
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

    test_dataset = LangDistillDataset(sentence_words=test_words,  
                                  int2vocab=test_index2vocab,
                                    w2v_vectors=w2v_emb, 
                                    tokenizer=tokenizer, 
                                    neg_seq_len=hparam['neg_seq_len'], top_k_negatives=3)

    total_iteration = hparam['max_epochs'] * len(train_dataset) // hparam['batch_size']
    hparam['total_iteration'] = total_iteration
    train_dataloader = DataLoader(
        train_dataset, 
        num_workers=(mp.cpu_count() - 1), 
        pin_memory=True,
        shuffle=True,  
        batch_size=hparam['batch_size'])
    test_dataloader = DataLoader(
        test_dataset, batch_size=hparam['batch_size'], num_workers=(mp.cpu_count() - 1), pin_memory=True, shuffle=False)

    config = DistillEmbConfig(
        num_input_chars=tokenizer.max_word_length,  # number of characters in each token
        char_vocab_size=tokenizer.char_vocab_size,
        size=hparam['size'],
        distill_dropout=hparam['dropout'],
        use_normalize=hparam['use_normalize'],
        use_tanh=hparam['use_tanh'],
        activation=hparam['activation'] 
    )
    distill_emb = DistillEmb(config)
    

    cbs = []

    logger = WandbLogger(log_model=True, 
                    save_dir="logs/", 
                    name=hparam['run_name'], 
                    project="distill_emb", 
                    entity="leobitz")

    
    # cbs.append(checkpoint_callback)
    trainer = L.Trainer(
                max_epochs=hparam['max_epochs'], 
                logger=logger, 
                log_every_n_steps=1,
                gradient_clip_val=hparam['clip_grad_norm'],
                gradient_clip_algorithm='norm', 
                callbacks=cbs)

    trainer.fit(model=DistillModule(model=distill_emb, tokenizer=tokenizer, **hparam),
                train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    # After training, optionally push the model to Hugging Face Hub
    # Requires: `huggingface_hub` installed and HF authentication done (`huggingface-cli login`)
    repo_id = hparam.get("hf_repo_id", None)
    if repo_id is not None:
        try:

            api = HfApi()
            create_repo(
                repo_id=repo_id,
                private=True,
                exist_ok=True,
            )

            # Save model and tokenizer locally in a temp directory
            save_dir = trainer.model.best_task_performance['path']
            

            api.upload_folder(
                folder_path=save_dir,
                repo_id=repo_id,
            )
            print(f"Pushed model to Hugging Face Hub: {repo_id}")
        except Exception as e:
            print(f"Failed to push model to Hugging Face Hub: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--use_tanh", action="store_true")
    parser.add_argument("--use_normalize", action="store_true")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--vector_load_ratio", type=float, default=0.5)
    parser.add_argument("--sentence_load_ratio", type=float, default=0.5)
    parser.add_argument("--min_sent_length", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--neg_seq_len", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--config_json", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--max_word_piece", type=int, default=10)
    # run_name
    parser.add_argument("--run_name", type=str, default=None)
    # task_eval_every
    parser.add_argument("--task_eval_every", type=int, default=4)
    # repo id
    parser.add_argument("--hf_repo_id", type=str, default=None)
    # size
    parser.add_argument("--size", type=str, default=None)
    # activation
    parser.add_argument("--activation", type=str, default='relu')

    args = parser.parse_args()
    hparam = vars(args)

    if args.config_json is not None:
        with open(args.config_json, "r") as f:
            cfg = json.load(f)
        if "train" in cfg:
            for k, v in cfg["train"].items():
                hparam.setdefault(k, v)
        if "model" in cfg:
            for k, v in cfg["model"].items():
                hparam.setdefault(k, v)

    main(hparam)
# sample usage: train_distill.py --max_epochs 50 --batch_size 256 --lr 0.001 --weight_decay 0.01 --dropout 0.1 --normalize --use_tanh --clip_grad_norm 1.0 --vector_load_ratio 0.5 --sentence_load_ratio 0.5 --min_sent_length 3 --train_ratio 0.8 --neg_seq_len 16 --temperature 0.1