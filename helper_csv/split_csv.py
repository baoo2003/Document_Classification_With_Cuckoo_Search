import os
import math
import pandas as pd

# ===== Cấu hình =====
input_csv = "dataset/data_dantri.csv"
out_dir = "dataset/chunks4"
target_link = "https://dantri.com.vn/the-thao/hoang-xuan-vinh-gap-lai-doi-thu-goc-trung-quoc-201608101359262.htm"
n_parts = 2

os.makedirs(out_dir, exist_ok=True)

# ===== Đọc dữ liệu =====
df = pd.read_csv(input_csv)

# ===== Tìm vị trí dòng có link =====
matches = df.index[df["link"] == target_link].tolist()

if not matches:
    raise ValueError(f"❌ Không tìm thấy link: {target_link}")

start_idx = matches[0]   # lấy vị trí đầu tiên (nếu hiếm khi trùng)
print(f"✅ Found link at row index: {start_idx}")

# ===== Lấy dữ liệu từ dòng đó trở đi (BAO GỒM) =====
df_tail = df.iloc[start_idx:].reset_index(drop=True)

total_rows = len(df_tail)
part_size = math.ceil(total_rows / n_parts)

print(f"Total rows from target link: {total_rows}")
print(f"Each part ~ {part_size} rows")

# ===== Chia thành 5 tệp (CÓ HEADER) =====
for i in range(n_parts):
    start = i * part_size
    end = min((i + 1) * part_size, total_rows)

    chunk = df_tail.iloc[start:end]

    out_path = (
        f"{out_dir}/chunk_{i+1}_rows_"
        f"{start_idx+start}_{start_idx+end-1}.csv"
    )

    chunk.to_csv(out_path, index=False, header=True)

    print(f"Saved {out_path} ({len(chunk)} rows)")
