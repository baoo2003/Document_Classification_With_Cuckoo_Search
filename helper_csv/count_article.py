# import pandas as pd

# # ===== 1. Đọc dữ liệu =====
# df = pd.read_csv("dataset/UIT-ViON_val_preprocessed.csv")

# # # ===== 2. Lọc bài báo Dan Tri =====
# df_vne = df[df["link"].str.contains("dantri.com.vn", na=False)]

# # ===== 3. Đếm số lượng mỗi label =====
# label_counts = df_vne["label"].value_counts()
# min_count = label_counts.min()

# print("Số lượng nhỏ nhất mỗi label:", min_count)
# print(label_counts)

# # ===== 4. Cân bằng nhãn (random undersampling) =====
# balanced_df = (
#     df_vne
#     .groupby("label", group_keys=False)
#     .apply(lambda x: x.sample(n=min_count, random_state=42))
# )

# # ===== 5. Shuffle lại toàn bộ dataset =====
# balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # ===== 6. Lưu file mới =====
# balanced_df.to_csv("dataset/data_dantri_balanced_val.csv", index=False)

# print("✅ Đã lưu file data_dantri_balanced_val.csv")
# print("Số lượng mỗi label sau khi cân bằng:")
# print(balanced_df["label"].value_counts())


import pandas as pd

# ===== 1. Đọc dữ liệu =====
df = pd.read_csv("dataset/UIT-ViON_test_preprocessed.csv")

# ===== 2. Lọc bài báo Vietnamnet =====
df_vnn = df[df["link"].str.contains("vnexpress.net", na=False)].copy()

# ===== 3. Chỉ lấy label 6 và 8 =====
df_vnn = df_vnn[df_vnn["label"].isin([0,1,2,3,4,5,7,9,10,11,12])]

# ===== 4. Kiểm tra số lượng hiện có =====
label_counts = df_vnn["label"].value_counts()
print("Số lượng hiện có mỗi label:")
print(label_counts)

TARGET_COUNT = 10

# ===== 5. Random lấy đúng 2963 cho mỗi label =====
balanced_df = (
    df_vnn
    .groupby("label", group_keys=False)
    .apply(lambda x: x.sample(
        n=TARGET_COUNT if len(x) >= TARGET_COUNT else len(x),
        random_state=42
    ))
)

# ===== 6. Shuffle lại dataset =====
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ===== 7. Lưu file mới =====
balanced_df.to_csv("dataset/data_vnexpress_label_10_test.csv", index=False)

print("✅ Đã lưu file dataset/data_vnexpress_label_10_test.csv")
print("Số lượng sau khi lấy mẫu:")
print(balanced_df["label"].value_counts())
