from torch.utils.data import Dataset
import numpy as np
import random
import re
from tokenizer import CharTokenizer as Tokenizer
import torch
import json
import os

def load_word_embeddings(file_path: str, target_words: set = None, header: bool = True, word_prob=1.0) -> dict:
    word2vec = {}
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        if header:
            line = f.readline()
            n_vecs, dim = int(line.split(' ')[0]), int(line.split(' ')[1])
        count = 0
        for line in f:
            if random.random() < word_prob:
                line = line.strip().split(' ')
                word = line[0]
                vec = line[1:]
                if (target_words == None or word in target_words) and len(vec) == dim:
                    word2vec[word] = np.array([float(x) for x in vec], dtype=np.float32)
            count += 1
        n_vecs = count
    print(f"Embed size: {len(word2vec[word])} Total Embeddings : {len(word2vec)}/{n_vecs}")
    return word2vec


def load_corpus_words(path, vocab_set, line_prob=1.0, min_sent_length=8, vocab2lang_file=None, lang2word_file=None):
    if vocab2lang_file is not None and lang2word_file is not None:
        if os.path.isfile(lang2word_file) and os.path.isfile(vocab2lang_file):
            vocab2lang = json.load(open(vocab2lang_file, 'r', encoding='utf-8'))
            all_vocabs = json.load(open(lang2word_file, 'r', encoding='utf-8'))
            return all_vocabs, vocab2lang
    
    all_words = {}
    vocab2lang = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if random.random() < line_prob:
                line = line.strip()
                lang, text = line.split('\t')
                
                if lang not in all_words:
                    all_words[lang] = set()
                    
                words = text.split()
                words = [word for word in words if word in vocab_set]
                all_words[lang] |= set(words)
                
                for word in words:
                    if word not in vocab2lang:
                        vocab2lang[word] = {}
                            
                    if bool(re.match(r'^[\d\W]+$', word)):
                        if 'punc' not in vocab2lang[word]:
                            vocab2lang[word]['punc'] = 0
                        vocab2lang[word]['punc'] += 1
                    else:
                        if lang not in vocab2lang[word]:
                            vocab2lang[word][lang] = 0
                        vocab2lang[word][lang] += 1
    
    # select the language for each vocab with highest count
    for word in vocab2lang:
        langs = vocab2lang[word]
        lang = max(langs, key=langs.get)
        vocab2lang[word] = lang
    for lang in all_words:
        all_words[lang] = list(all_words[lang])
    if lang2word_file is not None and vocab2lang_file is not None:
        json.dump(vocab2lang, open(vocab2lang_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        json.dump(all_words, open(lang2word_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    return all_words, vocab2lang


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
        # apply tanh to pos_w2v and neg_w2v
        return target_chars, pos_w2v, neg_w2v
