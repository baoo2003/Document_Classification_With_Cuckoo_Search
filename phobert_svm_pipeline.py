import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# =============================
# 1) PhoBERT utilities + Loader
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_phobert(local_dir: str = "models/phobert-base",
                 model_name: str = "vinai/phobert-base"):
    """
    Nếu local_dir tồn tại -> load PhoBERT từ local.
    Nếu không -> tải từ Hugging Face và lưu vào local_dir.
    """
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f">> Loading PhoBERT from local: {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=False)
        model = AutoModel.from_pretrained(local_dir).to(device).eval()
    else:
        print(">> Downloading PhoBERT from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=local_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=local_dir).to(device).eval()
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        print(f">> Saved PhoBERT to {local_dir}")
    return tokenizer, model

# Khởi tạo dùng chung
_tokenizer, _model = load_phobert()

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B,1]
    return summed / counts

def phobert_embed(texts, max_length: int = 256, batch_size: int = 16, l2norm: bool = True) -> np.ndarray:
    """
    Trả về embedding PhoBERT (numpy) cho list[str].
    Không embed nếu danh sách rỗng.
    """
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)

    embs = []
    _model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = _tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            ).to(device)

            outputs = _model(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            if l2norm:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            embs.append(pooled.cpu().numpy())
    return np.vstack(embs)

# =============================
# 2) Train + Evaluate + Save
# =============================
def train_phobert_svm(csv_path: str,
                      title_col: str = "title",
                      content_col: str = "content",
                      label_col: str = "topic",
                      test_size: float = 0.2,
                      random_state: int = 42,
                      batch_size: int = 8,
                      max_length: int = 256,
                      C: float = 1.0,
                      save_dir: str = "phobert_svm_model"):
    """
    Train PhoBERT embeddings + LinearSVC.
    Lưu model, label encoder, test set (text), và train/test embeddings.
    Trả về: clf, le, X_test_txt, y_test (y_test dạng int).
    """

    print(f"Run on: {device}")

    # Load data
    df = pd.read_csv(csv_path)
    if not {title_col, content_col, label_col}.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {title_col}, {content_col}, {label_col}")

    df["text"] = df[title_col].fillna("") + " " + df[content_col].fillna("")
    texts = df["text"].astype(str).tolist()
    labels_str = df[label_col].astype(str).tolist()

    # Encode label
    le = LabelEncoder()
    y = le.fit_transform(labels_str)

    # Split
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Embed & Train
    print(">> Embedding train set...")
    X_train = phobert_embed(X_train_txt, batch_size=batch_size, max_length=max_length)
    print(">> Embedding test set...")
    X_test = phobert_embed(X_test_txt, batch_size=batch_size, max_length=max_length)

    print(">> Training SVM...")
    clf = LinearSVC(C=C, max_iter=5000)
    clf.fit(X_train, y_train)

    # Evaluate quick
    y_pred = clf.predict(X_test)
    print(">> Evaluation:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(clf, f"{save_dir}/svm_model.joblib")
    joblib.dump(le, f"{save_dir}/label_encoder.joblib")
    joblib.dump((X_test_txt, y_test), f"{save_dir}/test_set.pkl")

    np.save(f"{save_dir}/X_train_emb.npy", X_train)
    np.save(f"{save_dir}/y_train.npy", y_train)
    np.save(f"{save_dir}/X_test_emb.npy",  X_test)
    np.save(f"{save_dir}/y_test.npy",      y_test)

    print(f">> Saved model, encoder, test set and embeddings to {save_dir}/")
    return clf, le, X_test_txt, y_test

# =============================
# 3) Load + Predict tiện dụng
# =============================
def load_model(save_dir: str = "phobert_svm_model"):
    clf = joblib.load(f"{save_dir}/svm_model.joblib")
    le = joblib.load(f"{save_dir}/label_encoder.joblib")
    return clf, le

def load_test_set(save_dir: str = "phobert_svm_model"):
    """Load lại X_test_txt, y_test (int) đã lưu lúc train."""
    return joblib.load(f"{save_dir}/test_set.pkl")

def load_embeddings(save_dir: str = "phobert_svm_model"):
    X_train = np.load(f"{save_dir}/X_train_emb.npy")
    y_train = np.load(f"{save_dir}/y_train.npy")
    X_test  = np.load(f"{save_dir}/X_test_emb.npy")
    y_test  = np.load(f"{save_dir}/y_test.npy")
    return X_train, y_train, X_test, y_test

def predict_topic(title: str, content: str, clf, le,
                  batch_size: int = 8, max_length: int = 256) -> str:
    text = (title or "") + " " + (content or "")
    vec = phobert_embed([text], batch_size=batch_size, max_length=max_length)
    pred = clf.predict(vec)[0]
    return le.inverse_transform([pred])[0]

def predict_title_content_batch(titles, contents, clf, le,
                                batch_size: int = 8, max_length: int = 256):
    texts = [(t or "") + " " + (c or "") for t, c in zip(titles, contents)]
    X = phobert_embed(texts, batch_size=batch_size, max_length=max_length)
    preds = clf.predict(X)
    return le.inverse_transform(preds).tolist()

def _ensure_embeddings(X_emb=None, X_texts=None, batch_size: int = 8, max_length: int = 256):
    """
    Nếu đã có X_emb (ndarray) -> dùng ngay.
    Nếu chỉ có X_texts (list[str]) -> embed tại đây.
    """
    if X_emb is not None:
        return X_emb
    if X_texts is None:
        raise ValueError("Cần truyền X_emb hoặc X_texts.")
    return phobert_embed(X_texts, batch_size=batch_size, max_length=max_length)

# =============================
# 4) Visualization (matplotlib)
# =============================
def evaluate_confusion_matrix(
    clf, le,
    X_emb=None,               # ưu tiên: embedding đã lưu
    X_texts=None,             # fallback: text (sẽ embed)
    y_true_labels=None,       # y_true có thể là int hoặc tên lớp
    batch_size: int = 8, max_length: int = 256,
    normalize: bool = False, figsize=(10, 8),
    title: str = "Confusion Matrix"
):
    if y_true_labels is None:
        raise ValueError("Cần truyền y_true_labels (int hoặc tên lớp).")

    # Chuẩn hóa y_true
    if isinstance(y_true_labels[0], str):
        y_true = le.transform(y_true_labels)
    else:
        y_true = np.asarray(y_true_labels)

    X = _ensure_embeddings(X_emb=X_emb, X_texts=X_texts,
                           batch_size=batch_size, max_length=max_length)

    y_pred = clf.predict(X)
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format=".2f" if normalize else "d")
    ax.set_title(title + (" (normalized)" if normalize else ""))
    plt.tight_layout()
    plt.show()

def _prepare_vis_samples(texts, labels, le, max_points: int = 2000, random_state: int = 42):
    """
    Lấy mẫu gọn để vẽ (tránh quá nặng).
    Trả về: texts_s, y_s (int).
    """
    rng = np.random.RandomState(random_state)
    n = len(texts)
    idx = np.arange(n)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
    texts_s = [texts[i] for i in idx]

    # labels có thể là int hoặc str; convert về int theo le
    if isinstance(labels[0], str):
        y_s = le.transform([labels[i] for i in idx])
    else:
        y_s = np.asarray([labels[i] for i in idx], dtype=int)
    return texts_s, y_s

def plot_umap_embeddings(
    texts=None, labels=None, le=None,
    X_emb=None, y_int=None,                    # dùng cặp này nếu có sẵn
    batch_size: int = 8, max_length: int = 256,
    max_points: int = 4000, n_neighbors: int = 15, min_dist: float = 0.1,
    figsize=(8, 6), random_state: int = 42, title: str = "UMAP of PhoBERT embeddings"
):
    try:
        import umap
    except Exception as e:
        raise RuntimeError("UMAP chưa được cài. Chạy: pip install umap-learn") from e

    # Nếu đã có embedding & y_int: ưu tiên dùng trực tiếp (và xuống mẫu nếu quá lớn)
    if X_emb is not None and y_int is not None:
        X_all = X_emb
        y_all = np.asarray(y_int, dtype=int)
        n = len(y_all)
        rng = np.random.RandomState(random_state)
        idx = np.arange(n)
        if n > max_points:
            idx = rng.choice(n, size=max_points, replace=False)
        X = X_all[idx]
        y_s = y_all[idx]
    else:
        # Dùng texts/labels + embed tại đây
        if texts is None or labels is None or le is None:
            raise ValueError("Cần (texts, labels, le) hoặc (X_emb, y_int).")
        texts_s, y_s = _prepare_vis_samples(texts, labels, le, max_points, random_state)
        X = phobert_embed(texts_s, batch_size=batch_size, max_length=max_length)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=2,
        random_state=random_state, metric="cosine",
    )
    coords = reducer.fit_transform(X)

    plt.figure(figsize=figsize)
    classes = np.unique(y_s)
    for cls in classes:
        m = (y_s == cls)
        label = le.inverse_transform([cls])[0] if le is not None else str(cls)
        plt.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.7, label=label)
    plt.legend(loc="best", fontsize=8, markerscale=2)
    plt.title(title)
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.tight_layout(); plt.show()

def plot_learning_curve_svm(
    clf, X_emb=None, X_texts=None, y_labels=None,
    batch_size: int = 8, max_length: int = 256,
    cv: int = 5, train_sizes=np.linspace(0.1, 1.0, 5),
    scoring: str = "f1_macro", figsize=(8, 6),
    title: str = "Learning Curve (SVM on PhoBERT embeddings)"
):
    if y_labels is None:
        raise ValueError("Cần y_labels (int).")
    if isinstance(y_labels[0], str):
        raise ValueError("y_labels đang là str. Hãy encode trước bằng LabelEncoder.")

    X = _ensure_embeddings(X_emb=X_emb, X_texts=X_texts,
                           batch_size=batch_size, max_length=max_length)

    ts_abs, train_scores, valid_scores = learning_curve(
        estimator=clf, X=X, y=np.asarray(y_labels), cv=cv,
        scoring=scoring, train_sizes=train_sizes, n_jobs=-1
    )

    tr_m, tr_s = train_scores.mean(axis=1), train_scores.std(axis=1)
    va_m, va_s = valid_scores.mean(axis=1), valid_scores.std(axis=1)

    plt.figure(figsize=figsize)
    plt.plot(ts_abs, tr_m, "o-", label="Train score")
    plt.fill_between(ts_abs, tr_m - tr_s, tr_m + tr_s, alpha=0.15)
    plt.plot(ts_abs, va_m, "o-", label="CV score")
    plt.fill_between(ts_abs, va_m - va_s, va_m + va_s, alpha=0.15)
    plt.xlabel("Training examples"); plt.ylabel(scoring); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()
