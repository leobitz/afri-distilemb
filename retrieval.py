# pip install transformers torch pandas tqdm scikit-learn --quiet

import random, json, torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


def mean_pool(last_hidden_state, attention_mask):
    """
    Standard mean-pooling for sentence embeddings.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, 1)
    counted = torch.clamp(mask.sum(1), min=1e-9)
    return summed / counted


class HFEncoder:
    """
    Light wrapper that turns any HuggingFace model into
    an `.encode()` function returning numpy vectors.
    """
    def __init__(self, model, tokenizer, model_name: str=None, pipeline: callable = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline
        if self.pipeline is None:
            if model is None or tokenizer is None:
                self.tok = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
            else:
                self.tok = tokenizer
                self.model = model.to(self.device).eval()
            self.pipeline = self.embed
        else:
            self.tok = None
            self.model = None

    def embed(self, batch: list[str]):
        enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                max_length=512,
            )
        # print({k:v.shape for k, v in enc.items()})
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            result = self.model(**enc)
        if type(result) is torch.Tensor:
            emb = result
        else:
            emb = result.last_hidden_state
        
        emb = mean_pool(emb, enc["attention_mask"])
        return emb.cpu().numpy()

    @torch.inference_mode()
    def encode(self, sentences: list[str], batch_size: int = 32) -> np.ndarray:
        outs = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            embedding = self.pipeline(batch)
            outs.append(embedding)
        return np.vstack(outs)


def build_json_pairs(
    df: pd.DataFrame,
    model_name: str = "bert-base-uncased",
    text_col: str = "text",
    headline_col: str = "headline",
    n_samples: int = 100,
    m_candidates: int = 30,
    k_top: int = 3,
    seed: int | None = None,
    keep_correct_in_pool: bool = False,

) -> list[dict]:
    """
    Parameters
    ----------
    df : dataframe with the two columns
    text_col, headline_col : which column is the *query* text and which is the pool
                             (swap them to support headline→text mode)
    n_samples : N  – how many rows to sample from the dataframe
    m_candidates : M – number of candidate headlines to encode per query
    k_top : K   – how many of the most-similar to keep (K+1 incl. the correct one)
    keep_correct_in_pool : if True, the correct headline may also enter the K pool.
                           If False (default) we exclude it before similarity ranking.

    Returns
    -------
    List of dicts: [{<query_text>: [gold_headline, distractor1, ... distractorK]}]
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    outputs = []
    for lang in df['lang'].unique():
        # 1) Randomly sample N rows
        land_df = df[df['lang'] == lang]
        sampled = land_df.sample(n=min(n_samples, len(land_df)), random_state=seed).reset_index(drop=True)

        # 2) Instantiate encoder
        encoder = HFEncoder(model_name)

        
        all_candidates = land_df[headline_col].tolist()

        for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Processing"):
            query_text = row[text_col]
            gold_headline = row[headline_col]

            # 3) Draw M random candidate headlines (optionally exclude the gold one)
            pool = random.sample(all_candidates, k=min(m_candidates, len(all_candidates)))
            if not keep_correct_in_pool:
                pool = [h for h in pool if h != gold_headline]
                # If we removed it and pool got too small, top up
                while len(pool) < m_candidates:
                    cand = random.choice(all_candidates)
                    if cand != gold_headline:
                        pool.append(cand)

            # 4) Encode query and the candidate pool
            embeddings = encoder.encode([query_text] + pool)
            query_emb, cand_embs = embeddings[0], embeddings[1:]
            # apply tanh to embeddings
            query_emb = np.tanh(query_emb)
            cand_embs = np.tanh(cand_embs)
            # 5) Pick K most-similar headlines
            sims = cosine_similarity([query_emb], cand_embs)[0]
            top_idx = sims.argsort()[::-1][:k_top]
            top_headlines = [pool[i] for i in top_idx]

            # 6) Insert the *correct* headline at index 0
            final_list = [gold_headline] + top_headlines
            outputs.append({
                "text": query_text,
                "headlines": final_list,
                "lang": lang
            })

    return outputs


# --------------------------- Evaluation routine ---------------------------- #
def top1_accuracy(
    qa_pairs: list[dict],
    batch_size: int = 32,
    model: AutoModel | None = None,
    tokenizer: AutoTokenizer | None = None,
    pipeline = None
) -> float:
    """
    Parameters
    ----------
    qa_pairs : output of the *data-building* step.
               Each item looks like {"query text": [gold_headline, distr1, …, distrK]}
    model_name : HF encoder to test.
    batch_size : batching for faster GPU inference.

    Returns
    -------
    accuracy ∈ [0,1] : proportion of queries where argmax-similarity == 0.
    """
    " qa_pairs has text, headlines and lang keys "
    encoder = HFEncoder(model=model, tokenizer=tokenizer, pipeline=pipeline)
    correct = total = 0
    lang_acc = {}
    lang_count = {}
    for obj in tqdm(qa_pairs, desc="Evaluating"):
        query_text = obj["text"]
        candidates = obj["headlines"]  # [gold_headline, distractor1,
        lang = obj["lang"]
        # get embeddings
        query_emb = encoder.encode([query_text])[0]          # (d,)
        cand_embs = encoder.encode(candidates, batch_size)   # (K+1, d)

        # cosine similarities
        sims = cosine_similarity([query_emb], cand_embs)[0]  # (K+1,)
        pred = sims.argmax()                                 # index of best headline
        # print(f"predicted: {pred}, gold: 0, lang: {lang}, sim: {sims[pred]:.4f}")
        correct += pred == 0   # gold headline is at index 0
        total += 1
        lang_acc[lang] = lang_acc.get(lang, 0) + (pred == 0)
        lang_count[lang] = lang_count.get(lang, 0) + 1

    per_lang = {k:v / lang_count[k] for k, v in lang_acc.items()}
    lang_average = sum(per_lang.values()) / len(per_lang) 
    return correct / total, lang_average, per_lang



if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        "text": ["This is a query text", "Another query here"],
        "headline": ["Correct headline 1", "Correct headline 2"]
    })
    pairs = build_json_pairs(df, model_name="sentence-transformers/all-MiniLM-L6-v2",
                             n_samples=2, m_candidates=5, k_top=3)
    print(json.dumps(pairs, ensure_ascii=False, indent=2))