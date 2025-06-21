import cv2
import os

# Pfade definieren
video_folder = "aufnahmen"
snapshot_folder = "snapshots"
os.makedirs(snapshot_folder, exist_ok=True)

# Alle .avi-Dateien im Ordner durchgehen
video_files = [f for f in os.listdir(video_folder) if f.endswith(".avi")]

print(f"📹 {len(video_files)} Videos gefunden. Starte Verarbeitung...")

for filename in video_files:
    video_path = os.path.join(video_folder, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Konnte {filename} nicht öffnen.")
        continue

    # Letztes Frame ermitteln
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()

    if ret:
        snapshot_name = os.path.splitext(filename)[0] + ".jpg"
        snapshot_path = os.path.join(snapshot_folder, snapshot_name)
        cv2.imwrite(snapshot_path, frame)
        print(f"✅ Snapshot gespeichert: {snapshot_path}")
    else:
        print(f"⚠️ Konnte letztes Frame von {filename} nicht lesen.")

    cap.release()

print("\n🎉 Alle Snapshots gespeichert in:", os.path.abspath(snapshot_folder))
