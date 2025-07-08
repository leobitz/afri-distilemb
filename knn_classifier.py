import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
import torch

class KNNTextClassifier:

    def __init__(self, tokenizer, model=None, device="cuda"):
        self.tokenizer = tokenizer
        self.model = model.eval().to(device)

    def classifiy(self,
            model,
            tokenizer,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            batch_size: int = 32,
            k: int = 5,
            metric: str = "cosine",
            n_jobs: int = -1,
            device: str = "cuda"
    ):
        """
        Train & evaluate a K-Nearest-Neighbors text classifier.

        Parameters
        ----------
        model : callable
            Function or callable object that takes a list/iterable of texts and
            returns a 2-D array-like of embeddings (shape: [batch_size, dim]).
        train_df, test_df : pd.DataFrame
            Each must contain exactly two columns: 'text' and 'label'.
        batch_size : int, default 32
            Number of texts to embed per forward pass.  Tune to fit GPU/CPU RAM.
        k : int, default 5
            Number of neighbors for KNN.
        metric : str, default "cosine"
            Distance metric recognised by scikit-learn (e.g. "euclidean", "cosine").
        n_jobs : int, default -1
            Parallelism for KNN queries (-1 → all cores).

        Returns
        -------
        f1 : float
            Weighted F1-score on the test set.
        acc : float
            Accuracy on the test set.
        """

        assert model is not None or self.model is not None, \
            "You must provide a model or set it in the KNNTextClassifier instance."

        if model is None:
            model = self.model
        if tokenizer is None:
            tokenizer = self.tokenizer

        def embed(texts: list) -> np.ndarray:
            """Embed a list of texts using the provided model and tokenizer."""
            tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            with torch.no_grad():
                batch_emb = model(**tokenized)
            
            # Ensure numpy float32 on CPU for sklearn
            batch_emb = (
                batch_emb.detach().cpu().numpy()  # if torch.Tensor
                if hasattr(batch_emb, "detach")
                else np.asarray(batch_emb)
            ).astype(np.float32).mean(axis=1)  # average over tokens if needed
            return batch_emb

        def _batch_embed(text_series: pd.Series) -> np.ndarray:
            """Vectorise texts in batches to avoid GPU / RAM spikes."""
            embeds = []
            for start in range(0, len(text_series), batch_size):
                batch_texts = text_series.iloc[start:start + batch_size].tolist()
                batch_emb = embed(batch_texts)

                embeds.append(batch_emb)
            return np.vstack(embeds)

        # ---------- 1. Embed & fit ----------
        X_train = _batch_embed(train_df["text"])
        y_train = train_df["label"].values  # string or int labels both fine

        knn = KNeighborsClassifier(
            n_neighbors=k, metric=metric, n_jobs=n_jobs
        )
        knn.fit(X_train, y_train)

        # ---------- 2. Embed test & predict ----------
        X_test = _batch_embed(test_df["text"])
        y_test = test_df["label"].values
        y_pred = knn.predict(X_test)
        test_df['pred'] = y_pred

        
        per_lang = {}
        # ---------- 3. Metrics ----------
        if 'lang' in test_df.columns:
            total_samples = len(y_test)
            weighted_f1 = 0.0
            weighted_acc = 0.0
            
            for lang in test_df['lang'].unique():
                lang_idx = test_df['lang'] == lang
                lang_weight = np.sum(lang_idx) / total_samples
                f1_lang = f1_score(y_test[lang_idx], y_pred[lang_idx], average="weighted")
                acc_lang = accuracy_score(y_test[lang_idx], y_pred[lang_idx])
                weighted_f1 += f1_lang * lang_weight
                weighted_acc += acc_lang * lang_weight
                per_lang[lang] = {
                    'f1': f1_lang,
                    'acc': acc_lang
                }
            f1 = weighted_f1
            acc = weighted_acc
        else:
            f1 = f1_score(y_test, y_pred, average="weighted")
            acc = accuracy_score(y_test, y_pred)
        return f1, acc, per_lang, test_df
