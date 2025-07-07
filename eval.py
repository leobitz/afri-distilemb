import torch
import random
import numpy as np

def __rand_seq(words, max_len):
    if len(words) <= max_len:
        return words
    indexes = list(range(len(words)))
    random.shuffle(indexes)
    indexes = indexes[:max_len]
    indexes = sorted(indexes)
    return [words[i] for i in indexes]

def ir_eval(logger, lm, file_path, log_label, top=1):
    news_ir_dataset = NewsIRDataset(corpus_path=file_path, max_word_len=15)
    recalls = []
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)
    for di in range(len(news_ir_dataset.data)):
        headlines, articles = news_ir_dataset.data[di]
        
        headlines = [h.split() for h in headlines]
        head_rand_ids = list(range(len(headlines)))
        # random.shuffle(head_rand_ids)
        head_rand_ids.reverse() # so that target is at the bottom
        headlines = [headlines[i] for i in head_rand_ids] # shuffle headlines
        h_lens = [len(h) for h in headlines]
        
        articles = [__rand_seq(a.split(), 128) for a in articles]
        # randomly select words
        
        # articles = [random.sample(a, min(64, len(a))) for a in articles]
        art_rand_ids = list(range(len(articles)))
        # random.shuffle(art_rand_ids)
        art_rand_ids.reverse()
        articles = [articles[i] for i in art_rand_ids] # shuffle articles
        a_lens = [len(a) for a in articles]

        headline_embs = lm.encode_sentences(headlines) # shape = (batch, seq_len, hidden_size)
        article_embs = lm.encode_sentences(articles) # shape = (batch, seq_len, hidden_size)

        head_embs, art_embs = [], []
        for i, (xl, yl) in enumerate(zip(h_lens, a_lens)):
            hemb = headline_embs[i, :xl].mean(0) # shape = (hidden_size)
            aemb = article_embs[i, :yl].mean(0) # shape = (hidden_size)
            
            head_embs.append(hemb)
            art_embs.append(aemb)
            
        head_embs = torch.stack(head_embs) # shape = (batch, hidden_size)
        art_embs = torch.stack(art_embs) # shape = (batch, hidden_size)
        print(head_embs.shape, art_embs.shape)
        sim = cosine_similarity(head_embs[head_rand_ids.index(0)].unsqueeze(0), art_embs) # shape = (batch)
        sorted_sim, indices = sim.sort(descending=True) # sort similarity scores and get the indices
        pred_index = (indices == art_rand_ids.index(0)).nonzero().item() # get the index of the correct article
        head_recall =  (1 / (pred_index + 1)) # recall@k
        # subtract minimum recall
        if pred_index == len(articles) - 1:
            head_recall = 0
        
        sim = cosine_similarity(art_embs[art_rand_ids.index(0)].unsqueeze(0), head_embs) # shape = (batch)
        sorted_sim, indices = sim.sort(descending=True) # sort similarity scores and get the indices
        pred_index  = (indices == head_rand_ids.index(0)).nonzero().squeeze(1) # get the index of the correct article
        art_recall = (1 / (pred_index + 1)).item()  # recall@k
        if pred_index == len(headlines) - 1:
            art_recall = 0
        recalls.append((head_recall, art_recall)) # append recall@k
        

    recall = np.array(recalls).mean(0) # mean recall@k [headline, article]
    
    logger(log_label + "_headline_recall", recall[0])
    logger(log_label + "_article_recall", recall[1])
    
    return recall 
    