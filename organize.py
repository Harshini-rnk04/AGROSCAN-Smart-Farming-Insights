import os
import shutil
import pandas as pd
import json

# === Paths ===
base_dir = r"C:\Users\RAJA\Downloads\crop\Data"
csv_path = os.path.join(base_dir, "train.csv")
json_path = os.path.join(base_dir, "label_num_to_disease_map.json")
images_dir = os.path.join(base_dir, "train_images")
base1_dir =r"E:\AgroScan\dataset"
# === Output folders ===
healthy_dir = os.path.join(base1_dir, "healthy")
unhealthy_dir = os.path.join(base1_dir, "unhealthy")
os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(unhealthy_dir, exist_ok=True)

# === Load CSV and label mapping
df = pd.read_csv(csv_path)
with open(json_path, "r") as f:
    label_map = json.load(f)

# === Loop through each row and copy image
for _, row in df.iterrows():
    image_id = str(row["image_id"])
    label = str(row["label"])
    label_name = label_map.get(label)

    # Choose category based on label
    category = "healthy" if label_name.strip().lower() == "healthy" else "unhealthy"

    # File paths
    image_filename = image_id 
    src = os.path.join(images_dir, image_filename)
    dst = os.path.join(healthy_dir if category == "healthy" else unhealthy_dir, image_filename)

    # Copy if exists
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"❌ Image not found: {src}")
