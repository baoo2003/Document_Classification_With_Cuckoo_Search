import pandas as pd

# ===== 1. Đọc dữ liệu gốc =====
input_csv = "dataset/data_real_full.csv"

df = pd.read_csv(input_csv)

# ===== 2. Xác định description rỗng =====
mask_empty_desc = (
    df["description"].isna() |
    (df["description"].astype(str).str.strip() == "")
)

# ===== 3. Tách thành 2 DataFrame =====
df_missing_desc = df[mask_empty_desc].copy()
df_full_desc = df[~mask_empty_desc].copy()

# ===== 4. Lưu ra 2 file CSV mới =====
df_full_desc.to_csv(
    "dataset/data_real_full_description.csv",
    index=False
)

df_missing_desc.to_csv(
    "dataset/data_real_missing_description.csv",
    index=False
)

# ===== 5. Thông tin kiểm tra =====
print("✅ Tách file xong")
print("Full description:", len(df_full_desc))
print("Missing description:", len(df_missing_desc))
print("Total:", len(df))
