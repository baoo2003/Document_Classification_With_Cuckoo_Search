import os
import time
import csv
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

# =============================
# 1) Crawl 1 bài VNExpress
# =============================
def crawl_vnexpress_description(url: str, timeout: int = 30) -> str:
    """
    Chỉ lấy description:
    <p class="description"> (class duy nhất = description)
    Không cần nằm trong div nào.
    Trả về text 1 dòng (đã chuẩn hoá)
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    p_list = soup.select('p[class="description"]')
    p_desc = p_list[0] if p_list else None

    if not p_desc:
        return ""

    desc_text = p_desc.get_text(" ", strip=True)
    desc_text = " ".join(desc_text.split())  # gom whitespace
    return desc_text


# =============================
# 2) Crawl theo batch từ data.csv
# =============================
def enrich_dataset_vnexpress_description_only(
    input_csv="dataset/data_vnexpress_balanced.csv",
    output_csv="dataset/data_vnexpress_desc.csv",
    sleep_range=(0.2, 0.4),
    batch_size=20,
    resume=True
):
    df = pd.read_csv(input_csv)

    # chỉ crawl VNExpress
    df = df[df["link"].astype(str).str.contains("vnexpress.net", na=False)].copy()

    done_links = set()
    if resume and os.path.exists(output_csv):
        old = pd.read_csv(output_csv)
        done_links = set(old["link"].astype(str).tolist())

    todo = df[~df["link"].isin(done_links)].reset_index(drop=True)

    print(f"To crawl: {len(todo)} articles")

    file_exists = os.path.exists(output_csv)
    headers = ["title", "link", "label", "description"]

    buffer = []

    with open(output_csv, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writeheader()

        for i, row in todo.iterrows():
            url = row["link"]
            print(f"({i+1}/{len(todo)}) {url}")

            try:
                desc = crawl_vnexpress_description(url)
            except Exception as e:
                print(f"⚠️ Lỗi crawl: {e}")
                desc = ""

            buffer.append({
                "title": row["title"],
                "link": url,
                "label": row["label"],
                "description": desc
            })

            time.sleep(random.uniform(*sleep_range))

            if len(buffer) >= batch_size:
                writer.writerows(buffer)
                f.flush()
                buffer.clear()

        if buffer:
            writer.writerows(buffer)
            f.flush()

    print(f"✅ Done. Saved to {output_csv}")


if __name__ == "__main__":
    enrich_dataset_vnexpress_description_only(
        input_csv="dataset/data_vnexpress_balanced.csv",
        output_csv="dataset/data_vnexpress_desc.csv",
        sleep_range=(0.2, 0.4),
        batch_size=50,
        resume=True
    )