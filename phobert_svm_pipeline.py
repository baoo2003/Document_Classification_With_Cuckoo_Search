import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
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
def prepare_and_save_embeddings(
    csv_path, 
    title_col="title", content_col="content", label_col="topic",
    test_size=0.2, random_state=42, batch_size=8, max_length=256,
    save_dir="phobert_svm_model"
):
    texts, labels = load_data(csv_path, title_col, content_col, label_col)
    y, le = encode_labels(labels)
    X_train_txt, X_test_txt, y_train, y_test = split_data(texts, y, test_size, random_state)
    print(">> Embedding train set...")
    X_train = embed_texts(X_train_txt, batch_size, max_length)
    print(">> Embedding test set...")
    X_test = embed_texts(X_test_txt, batch_size, max_length)
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/X_train_emb.npy", X_train)
    np.save(f"{save_dir}/y_train.npy", y_train)
    np.save(f"{save_dir}/X_test_emb.npy",  X_test)
    np.save(f"{save_dir}/y_test.npy",      y_test)
    # Lưu csv có cả text và label
    df_train = pd.DataFrame(X_train)
    df_train.insert(0, "label", y_train)
    df_train.insert(0, "text", X_train_txt)    
    df_train.to_csv(f"{save_dir}/X_train_emb.csv", index=False)
    df_test = pd.DataFrame(X_test)
    df_test.insert(0, "label", y_test)
    df_test.insert(0, "text", X_test_txt)
    df_test.to_csv(f"{save_dir}/X_test_emb.csv", index=False)
    joblib.dump(le, f"{save_dir}/label_encoder.joblib")
    print(">> Saved embeddings and texts to", save_dir)

def train_svm_from_embeddings(
    save_dir="phobert_svm_model", 
    C=1.0, max_iter=5000
):
    X_train = np.load(f"{save_dir}/X_train_emb.npy")
    y_train = np.load(f"{save_dir}/y_train.npy")
    X_test  = np.load(f"{save_dir}/X_test_emb.npy")
    y_test  = np.load(f"{save_dir}/y_test.npy")
    le = joblib.load(f"{save_dir}/label_encoder.joblib")
    clf = LinearSVC(C=C, max_iter=max_iter)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(">> Evaluation:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    joblib.dump(clf, f"{save_dir}/svm_model.joblib")
    print(">> Saved SVM model to", save_dir)
    return clf, le, X_test, y_test

# =============================
# 3) Load + Predict tiện dụng
# =============================
def load_data(csv_path, title_col="title", content_col="content", label_col="topic"):
    df = pd.read_csv(csv_path)
    if not {title_col, content_col, label_col}.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {title_col}, {content_col}, {label_col}")
    df["text"] = df[title_col].fillna("") + " " + df[content_col].fillna("")
    texts = df["text"].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    return texts, labels

def encode_labels(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le

def split_data(texts, y, test_size=0.2, random_state=42):
    return train_test_split(texts, y, test_size=test_size, random_state=random_state, stratify=y)

def embed_texts(texts, batch_size=8, max_length=256):
    return phobert_embed(texts, batch_size=batch_size, max_length=max_length)

def train_svm(X_train, y_train, C=1.0, max_iter=5000):
    clf = LinearSVC(C=C, max_iter=max_iter)
    clf.fit(X_train, y_train)
    return clf

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

def plot_roc_multiclass(
    clf, le, X_emb=None, X_texts=None, y_true_labels=None,
    batch_size=8, max_length=256,
    figsize=(8,6), title="ROC (one-vs-rest)", max_classes=10
):
    if y_true_labels is None:
        raise ValueError("Cần y_true_labels (int hoặc tên lớp).")
    if isinstance(y_true_labels[0], str):
        y_true = le.transform(y_true_labels)
    else:
        y_true = np.asarray(y_true_labels, dtype=int)

    X = _ensure_embeddings(X_emb=X_emb, X_texts=X_texts,
                           batch_size=batch_size, max_length=max_length)

    scores = clf.decision_function(X)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T

    y_bin = label_binarize(y_true, classes=np.arange(scores.shape[1]))

    plt.figure(figsize=figsize)
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), scores.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro-average AUC = {auc_micro:.3f}")

    classes = np.arange(min(scores.shape[1], max_classes))
    for c in classes:
        fpr, tpr, _ = roc_curve(y_bin[:, c], scores[:, c])
        plt.plot(fpr, tpr, alpha=0.8, label=f"{le.classes_[c]} (AUC={auc(fpr,tpr):.3f})")

    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.show()