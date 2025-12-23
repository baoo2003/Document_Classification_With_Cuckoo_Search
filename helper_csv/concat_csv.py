import pandas as pd

# ===== 1. Đọc dữ liệu =====
df_vne = pd.read_csv("dataset/data_vnexpress_balanced.csv")
df_vnn = pd.read_csv("dataset/data_vietnamnet_label6_8_2963.csv")

# ===== 2. (An toàn) chỉ giữ label 6, 8 của Vietnamnet =====
df_vnn = df_vnn[df_vnn["label"].isin([6, 8])]

# ===== 3. Kiểm tra cấu trúc cột =====
assert list(df_vne.columns) == list(df_vnn.columns), "❌ Hai file không cùng cấu trúc cột"

# ===== 4. NỐI THEO THỨ TỰ (KHÔNG shuffle) =====
merged_df = pd.concat([df_vne, df_vnn], ignore_index=True)

# ===== 5. Lưu file mới =====
merged_df.to_csv(
    "dataset/data_vnexpress_vietnamnet.csv",
    index=False
)

print("✅ Đã tạo file data_vnexpress_vietnamnet.csv")
print("Số dòng VNExpress:", len(df_vne))
print("Số dòng Vietnamnet (6,8):", len(df_vnn))
print("Tổng số dòng:", len(merged_df))
