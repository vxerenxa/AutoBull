from ultralytics import YOLO
from pathlib import Path

# ✅ Pfad zu deinem trainierten Modell
model_path = 'runs/detect/train/weights/best.pt'

# ✅ Ordner mit neuen Bildern
input_dir = Path('Data/aufnahmen')

# ✅ Inferenz starten
model = YOLO(model_path)
model.predict(
    source=str(input_dir),
    save=True,
    save_txt=True,
    imgsz=640,
    conf=0.25
)

print("\n✅ Inferenz abgeschlossen!")
print("📁 Bilder mit Boxen: runs/detect/predict/")
print("📄 YOLO-Labels (.txt): runs/detect/predict/labels/")
