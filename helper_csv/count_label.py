import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("dataset/data_real_full_description_clean.csv")

print(df['label'].dtype)
# ===== Chỉ lấy các dòng có description rỗng =====
# Bao gồm cả NaN và chuỗi rỗng / toàn khoảng trắng
# df = df[
#     df["description"].isna() |
#     (df["description"].astype(str).str.strip() == "")
# ]

# Chỉ lấy label 6 và 8
df = df[df["label"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])]

# Mapping domain báo -> tên hiển thị
news_sources = {
    "vnexpress.net": "VnExpress",
    "vietnamnet.vn": "Vietnamnet",
    "baomoi.com": "Bao Moi Online",
    "dantri.com.vn": "Dan Tri Online",
    "laodong.vn": "Lao Dong Online",
    "tuoitre.vn": "TuoiTre Online",
}

# Hàm xác định nguồn báo từ link
def detect_source(link):
    for domain, name in news_sources.items():
        if domain in str(link):
            return name
    return "Other"

df["source"] = df["link"].apply(detect_source)

# Đếm số lượng theo source và label
result = (
    df.groupby(["source", "label"])
      .size()
      .unstack(fill_value=0)
      .sort_index()
)

print(result)
