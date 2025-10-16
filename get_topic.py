import pandas as pd

# ===== Hàm lọc theo keyword =====
def filter_by_keyword(df, keyword, column="topic"):
    # chỉ lấy những dòng mà topic BẮT ĐẦU bằng keyword (không phân biệt hoa thường)
    mask = df[column].astype(str).str.lower().str.startswith(keyword.lower())
    filtered = df[mask]
    counts = filtered[column].value_counts()

    print(f"📊 Các topic bắt đầu bằng '{keyword}':")
    if counts.empty:
        print("⚠️ Không tìm thấy dòng nào.\n")
    else:
        for topic, count in counts.items():
            print(f"{topic}: {count}")
        print(f"\n➡️ Tổng cộng {len(filtered)} dòng\n")

def count_by_topic(csv_path: str):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "topic" not in df.columns:
        raise ValueError("File CSV phải có cột 'topic'.")

    counts = df["topic"].value_counts().sort_index()
    print("📊 Số lượng bài theo topic:")
    for topic, count in counts.items():
        print(f"{topic}: {count}")

if __name__ == "__main__":
    # ===== Đường dẫn file =====
    # file_path = "thanhnien_articles_fixed.csv"  # đổi lại tên file thật
    # df = pd.read_csv(file_path)

    # keywords = [
    #     "kinh tế",
    #     "thế giới",
    #     "giáo dục",
    #     "sức khỏe",
    #     "đời sống",
    #     "công nghệ",
    #     "pháp luật",
    #     "xe",
    #     "thể thao",
    #     "văn hóa",
    #     "giải trí",
    #     "chính trị",
    #     "thời sự",
    #     "du lịch",
    # ]

    # for kw in keywords:
    #     filter_by_keyword(df, kw)

    count_by_topic("dataset/thanhnien_articles_normalized.csv")
