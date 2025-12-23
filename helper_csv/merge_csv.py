import os
import pandas as pd

# ===== Cấu hình =====
chunks_dir = "dataset/data_sample"
output_csv = "dataset/data_sample/data_130.csv"

# ===== Lấy danh sách file CSV (theo thứ tự tên file) =====
chunk_files = sorted([
    os.path.join(chunks_dir, f)
    for f in os.listdir(chunks_dir)
    if f.endswith(".csv")
])

if not chunk_files:
    raise ValueError("❌ Không tìm thấy file chunk nào")

print(f"Found {len(chunk_files)} chunk files")

# ===== Đọc & gộp =====
dfs = []
for f in chunk_files:
    print("Reading:", f)
    df = pd.read_csv(f)
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

# ===== (Tuỳ chọn) loại trùng link =====
# merged_df = merged_df.drop_duplicates(subset=["link"], keep="first")

# ===== Lưu file mới =====
merged_df.to_csv(output_csv, index=False)

print(f"✅ Done. Merged file saved to {output_csv}")
print("Total rows:", len(merged_df))
