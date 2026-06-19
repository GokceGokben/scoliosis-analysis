import cv2
import os
import shutil
from pathlib import Path

# --- 1. SETTINGS ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_IMG_DIR = BASE_DIR / 'Data' / 'Raw' / 'images'
RAW_LABEL_DIR = BASE_DIR / 'Data' / 'Raw' / 'labels'
PROC_IMG_DIR = BASE_DIR / 'Data' / 'processed' / 'images'
PROC_LABEL_DIR = BASE_DIR / 'Data' / 'processed' / 'labels'

TARGET_SIZE = (512, 512)
SPLITS = ['train', 'valid', 'test']

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def process_split(split):
    img_dir = RAW_IMG_DIR / split
    label_dir = RAW_LABEL_DIR / split
    out_img_dir = PROC_IMG_DIR / split
    out_label_dir = PROC_LABEL_DIR / split

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    for img_file in img_dir.iterdir():
        if img_file.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp'):
            continue

        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"ERROR: {img_file.name} Could not be loaded, skipping.")
            continue

        # --- 2. IMAGE PREPROCESSING ---
        resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        enhanced = clahe.apply(resized)
        cv2.imwrite(str(out_img_dir / img_file.name), enhanced)

        # --- 3. LABELS: YOLO copy as it is since they are already normalized ---
        label_file = label_dir / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.copy2(str(label_file), str(out_label_dir / label_file.name))
        else:
            print(f"WARNING: {label_file.name} could not be found.")

        print(f"OK: {split}/{img_file.name}")


def process_data():
    for split in SPLITS:
        print(f"\n--- {split.upper()} split is being processed ---")
        process_split(split)
    print("\nAll data processed and saved!")


if __name__ == "__main__":
    process_data()