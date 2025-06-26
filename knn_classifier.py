import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score

def knn_text_classifier(
        model,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        batch_size: int = 32,
        k: int = 5,
        metric: str = "cosine",
        n_jobs: int = -1,
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

    def _batch_embed(text_series: pd.Series) -> np.ndarray:
        """Vectorise texts in batches to avoid GPU / RAM spikes."""
        embeds = []
        for start in range(0, len(text_series), batch_size):
            batch_texts = text_series.iloc[start:start + batch_size].tolist()
            batch_emb = model(batch_texts)

            # Ensure numpy float32 on CPU for sklearn
            batch_emb = (
                batch_emb.detach().cpu().numpy()  # if torch.Tensor
                if hasattr(batch_emb, "detach")
                else np.asarray(batch_emb)
            ).astype(np.float32)

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

    # ---------- 3. Metrics ----------
    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    return f1, acc
