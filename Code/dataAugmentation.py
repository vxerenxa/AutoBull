import os
import cv2
import random
import shutil
import albumentations as A
from pathlib import Path

# -------------- CONFIG ----------------
SOURCE_IMG_DIR = Path("Data/snapshots")
SOURCE_LABEL_DIR = Path("Data/labels")
AUGMENTED_IMG_DIR = Path("Data/augmented/images")
AUGMENTED_LABEL_DIR = Path("Data/augmented/labels")
VARIANTS_PER_IMAGE = 3  # z. B. 3x neue Versionen pro Original
# --------------------------------------

# 📦 YOLO-Format: [class_id, x_center, y_center, width, height]
def read_yolo_labels(label_path):
    with open(label_path, "r") as f:
        return [list(map(float, line.strip().split())) for line in f.readlines()]

def write_yolo_labels(labels, path):
    with open(path, "w") as f:
        for l in labels:
            f.write(" ".join(map(str, l)) + "\n")

# 📦 Albumentations-Augs
transform = A.Compose([
    A.Rotate(limit=10, p=0.8),
    A.RandomBrightnessContrast(p=0.6),
    A.RandomScale(scale_limit=0.1, p=0.6),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.HueSaturationValue(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 🧼 Zielordner vorbereiten
shutil.rmtree(AUGMENTED_IMG_DIR, ignore_errors=True)
shutil.rmtree(AUGMENTED_LABEL_DIR, ignore_errors=True)
AUGMENTED_IMG_DIR.mkdir(parents=True, exist_ok=True)
AUGMENTED_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# 🔁 Durch alle Bilder iterieren
img_paths = sorted(SOURCE_IMG_DIR.glob("*.jpg"))

for img_path in img_paths:
    label_path = SOURCE_LABEL_DIR / (img_path.stem + ".txt")
    if not label_path.exists():
        print(f"⚠️  Kein Label für {img_path.name}, überspringe.")
        continue

    image = cv2.imread(str(img_path))
    height, width = image.shape[:2]
    labels = read_yolo_labels(label_path)

    for i in range(VARIANTS_PER_IMAGE):
        bboxes = [l[1:] for l in labels]
        class_labels = [int(l[0]) for l in labels]

        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        if not transformed["bboxes"]:
            continue  # falls Boxen durch Aug verloren gehen

        new_image = transformed["image"]
        new_bboxes = transformed["bboxes"]
        new_classes = transformed["class_labels"]

        # Speichern
        out_name = f"{img_path.stem}_aug{i}"
        out_img_path = AUGMENTED_IMG_DIR / f"{out_name}.jpg"
        out_label_path = AUGMENTED_LABEL_DIR / f"{out_name}.txt"

        cv2.imwrite(str(out_img_path), new_image)
        label_lines = [[new_classes[j]] + list(new_bboxes[j]) for j in range(len(new_bboxes))]
        write_yolo_labels(label_lines, out_label_path)

print("\n✅ Augmentierung abgeschlossen.")
print(f"📁 Neue Bilder: {AUGMENTED_IMG_DIR}")
print(f"📄 Neue Labels: {AUGMENTED_LABEL_DIR}")
