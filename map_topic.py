# -*- coding: utf-8 -*-
import pandas as pd

INPUT  = "dataset/thanhnien_articles_fixed.csv"
OUTPUT = "thanhnien_articles_topic_sampled.csv"
RANDOM_STATE = 42

# ========== mapping root ==========
ROOTS_SINGLE = ["kinh tế","thế giới","giáo dục","công nghệ","pháp luật","xe","thể thao","chính trị",
                "văn hóa","giải trí","sức khỏe","đời sống"]

def map_root(raw: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip(): return None
    t = raw.strip().lower()
    if t.startswith("thời sựpháp luật") or t.startswith("thoi suphap luat"): return "pháp luật"
    if t.startswith("thời sự") or t.startswith("thoi su"): return None
    for r in ROOTS_SINGLE:
        if t.startswith(r): return r
    return None

def sample_pool(df, root, n, pools, ptr):
    if root not in pools: return pd.DataFrame()
    start, end = ptr[root], min(ptr[root] + n, len(pools[root]))
    ptr[root] = end
    return pools[root].iloc[start:end].copy()

def sample_pair(df, r1, r2, target, new_topic, pools, ptr):
    if target <= 0: return pd.DataFrame()
    n1, n2 = (target + 1)//2, target//2  # chia 50–50, r1 nhận phần lẻ
    a = sample_pool(df, r1, n1, pools, ptr)
    b = sample_pool(df, r2, n2, pools, ptr)
    need = target - (len(a)+len(b))
    if need > 0:  # bù phần thiếu
        a = pd.concat([a, sample_pool(df, r1, need, pools, ptr)], ignore_index=True)
        need = target - len(a) - len(b)
        if need > 0:
            b = pd.concat([b, sample_pool(df, r2, need, pools, ptr)], ignore_index=True)
    merged = pd.concat([a,b], ignore_index=True)
    if not merged.empty: merged.loc[:, "topic"] = new_topic
    return merged

def run(soluong: int):
    # 1) đọc & map root
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    req_cols = ["id","url","topic","publisher","publishedDate","crawAt","thumbnail","title","description","content"]
    for c in req_cols:
        if c not in df.columns: raise ValueError(f"Thiếu cột {c} trong {INPUT}")
    df["root"] = df["topic"].apply(map_root)
    base = df[df["root"].notna()].copy()

    # 2) tạo pools đã shuffle theo root (reproducible)
    pools, ptr = {}, {}
    for r, g in base.groupby("root"):
        pools[r] = g.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        ptr[r] = 0

    # 3) lấy mẫu cho các topic cuối
    final_parts = []

    # single topics
    for final_name, root_name in [
        ("kinh tế","kinh tế"),
        ("thế giới","thế giới"),
        ("giáo dục","giáo dục"),
        ("công nghệ","công nghệ"),
        ("pháp luật","pháp luật"),
        ("xe","xe"),
        ("thể thao","thể thao"),
        ("chính trị","chính trị"),
    ]:
        chunk = sample_pool(base, root_name, soluong, pools, ptr)
        if not chunk.empty: chunk.loc[:, "topic"] = final_name
        final_parts.append(chunk)

    # merged pairs
    final_parts.append(sample_pair(base, "văn hóa","giải trí", soluong, "văn hóa giải trí", pools, ptr))
    final_parts.append(sample_pair(base, "sức khỏe","đời sống", soluong, "sức khỏe đời sống", pools, ptr))

    result = pd.concat([x for x in final_parts if x is not None and not x.empty], ignore_index=True)
    result = result[req_cols]
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig", quoting=1)
    print(f"✅ Saved {len(result)} rows → {OUTPUT}")

# ===== chạy =====
if __name__ == "__main__":
    run(soluong=788)  # đổi số lượng mong muốn cho mỗi topic ở đây
