import os
import cv2
import albumentations as A
from pathlib import Path
from tqdm import tqdm

# === Verzeichnisse ===
IMG_DIR = Path("./Data/snapshots")
LABEL_DIR = Path("./Data/bbox_labels")
OUTPUT_IMG_DIR = Path("./Data/augmented/snapshots")
OUTPUT_LABEL_DIR = Path("./Data/augmented/bbox_labels")

OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# === Augmentierungs-Pipeline ===
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# === Bounding Box Clipping Funktion ===
def clip_bbox_yolo(bbox):
    x, y, w, h = bbox
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]

# === Unterstützte Bildformate ===
SUPPORTED_IMG_EXTS = [".jpg", ".jpeg", ".png"]

# === Verarbeite alle Labels ===
for label_file in tqdm(os.listdir(LABEL_DIR), desc="Verarbeite Labels"):
    if not label_file.endswith(".txt"):
        continue

    label_path = LABEL_DIR / label_file
    base_name = label_file.replace(".txt", "")
    
    # Finde das zugehörige Bild
    image_path = None
    for ext in SUPPORTED_IMG_EXTS:
        candidate = IMG_DIR / f"{base_name}{ext}"
        if candidate.exists():
            image_path = candidate
            break

    if not image_path:
        print(f"❌ Kein Bild gefunden für: {label_file}")
        continue

    # Bild laden
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️ Bild konnte nicht geladen werden: {image_path}")
        continue

    # YOLO-Label laden
    with open(label_path, "r") as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        bboxes.append([float(x), float(y), float(w), float(h)])
        class_labels.append(cls)

    if not bboxes:
        continue

    # === Mehrfache Augmentierung pro Bild ===
    for i in range(3):  
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

            # Bounding Boxes clippen
            clipped_bboxes = []
            clipped_labels = []
            for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
                clipped = clip_bbox_yolo(bbox)
                if clipped:
                    clipped_bboxes.append(clipped)
                    clipped_labels.append(label)

            if not clipped_bboxes:
                continue

            # === Speichern ===
            out_img_name = f"{base_name}_aug{i}.jpg"
            out_label_name = f"{base_name}_aug{i}.txt"

            cv2.imwrite(str(OUTPUT_IMG_DIR / out_img_name), augmented['image'])

            with open(OUTPUT_LABEL_DIR / out_label_name, 'w') as f:
                for bbox, cls in zip(clipped_bboxes, clipped_labels):
                    x, y, w, h = bbox
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        except Exception as e:
            print(f"⚠️ Fehler bei {image_path.name} (Aug {i}): {e}")
