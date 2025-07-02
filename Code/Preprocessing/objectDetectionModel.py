import os
import random
import shutil
from ultralytics import YOLO
from pathlib import Path

#folder structure and basic settings
BASE_DIR = Path(__file__).resolve().parents[1]  
SOURCE_IMG_DIR = BASE_DIR / 'Data/snapshots'
SOURCE_LABEL_DIR = BASE_DIR / 'Data/labels'
DEST_DIR = BASE_DIR / 'Data/YOLO-ready'
INFER_DIR = BASE_DIR / 'Data/aufnahmen'
KORREKTUR_DIR = BASE_DIR / 'Data/korrektur_benoetigt'

CLASS_NAMES = ['pfeil']
SPLIT_RATIO = 0.8
EPOCHS = 50
IMG_SIZE = 640
MODEL_TYPE = 'yolov8n.pt'


# ensure only valid images with labels are processed
all_imgs = sorted([f for f in SOURCE_IMG_DIR.iterdir() if f.suffix.lower() in ['.jpg', '.png']])
valid_imgs = []

for img_path in all_imgs:
    label_path = SOURCE_LABEL_DIR / (img_path.stem + '.txt')
    if label_path.exists():
        valid_imgs.append(img_path)
    else:
        print(f"⚠️  Kein Label für {img_path.name} → wird übersprungen")

random.seed(42)
random.shuffle(valid_imgs)

#split images into training and validation sets
split_idx = int(len(valid_imgs) * SPLIT_RATIO)
train_imgs = valid_imgs[:split_idx]
val_imgs = valid_imgs[split_idx:]

# create folder structure for YOLO
for split in ['train', 'val']:
    (DEST_DIR / f'images/{split}').mkdir(parents=True, exist_ok=True)
    (DEST_DIR / f'labels/{split}').mkdir(parents=True, exist_ok=True)

# copy images and labels to the new structure
def copy_files(img_paths, split):
    for img_path in img_paths:
        label_path = SOURCE_LABEL_DIR / (img_path.stem + '.txt')
        target_img = DEST_DIR / f'images/{split}' / img_path.name
        target_label = DEST_DIR / f'labels/{split}' / label_path.name
        shutil.copy(img_path, target_img)
        shutil.copy(label_path, target_label)

copy_files(train_imgs, 'train')
copy_files(val_imgs, 'val')

# write dataset.yaml 
with open(DEST_DIR / 'dataset.yaml', 'w') as f:
    f.write(f"train: images/train\n")
    f.write(f"val: images/val\n")
    f.write(f"nc: {len(CLASS_NAMES)}\n")
    f.write(f"names: {CLASS_NAMES}\n")

# train the YOLO model
model = YOLO(MODEL_TYPE)
model.train(data=str(DEST_DIR / 'dataset.yaml'), epochs=EPOCHS, imgsz=IMG_SIZE)

# infer new images
INFER_DIR.mkdir(exist_ok=True)

model.predict(
    source=str(INFER_DIR),
    save=True,
    save_txt=True,
    imgsz=IMG_SIZE,
    conf=0.25
)

# predict labels and move failed predictions
PRED_LABEL_DIR = Path('runs/detect/predict/labels')
KORREKTUR_DIR.mkdir(exist_ok=True)
fehlgeschlagene = []

for img_file in INFER_DIR.glob("*.[jp][pn]g"):
    txt_file = PRED_LABEL_DIR / (img_file.stem + '.txt')
    if not txt_file.exists() or txt_file.stat().st_size == 0:
        fehlgeschlagene.append(img_file.name)
        shutil.copy(img_file, KORREKTUR_DIR / img_file.name)

# output
print("\n✅ Alles fertig!")
print("📂 Trainingsdaten unter:", DEST_DIR)
print("📁 Vorhersagen: runs/detect/predict/")
if fehlgeschlagene:
    print(f"\n⚠️  {len(fehlgeschlagene)} fehlerhafte Predictions wurden verschoben nach '{KORREKTUR_DIR}'")
else:
    print("\n🎉 Alle Vorhersagen erfolgreich.")
