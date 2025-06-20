import cv2
import os

# === KONFIGURATION ===
video_path = "aufnahmen/clip_1_cam0_20250620_161439.avi"
label_path = video_path.replace(".avi", ".txt")  # gleichnamige Datei mit .txt
class_names = {0: "Pfeil"}  # Beispiel-Klasse(n)

# === Bounding Boxes laden (YOLO-Format) ===
def load_yolo_labels(txt_file, frame_width, frame_height):
    boxes = []
    if not os.path.exists(txt_file):
        print(f"⚠️ Keine Label-Datei gefunden: {txt_file}")
        return boxes
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = map(float, parts)
            x1 = int((x - w / 2) * frame_width)
            y1 = int((y - h / 2) * frame_height)
            x2 = int((x + w / 2) * frame_width)
            y2 = int((y + h / 2) * frame_height)
            boxes.append((int(cls_id), x1, y1, x2, y2))
    return boxes

# === Video abspielen mit Bounding Boxes ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Fehler beim Öffnen des Videos.")
    exit()

ret, frame = cap.read()
if not ret:
    print("❌ Kein Frame gelesen.")
    exit()

frame_height, frame_width = frame.shape[:2]
boxes = load_yolo_labels(label_path, frame_width, frame_height)

print("▶️ Video mit Bounding Boxes – Drücke 'q' zum Beenden.")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Zurück zum ersten Frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Bounding Boxes einzeichnen
    for cls_id, x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = class_names.get(cls_id, str(cls_id))
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Bounding Boxes", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
