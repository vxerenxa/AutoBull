import cv2
import time
import os
import numpy as np

# Kamera-IDs
raw_camera_ids = [0, 2, 4]
camera_names = {}

# Kameraobjekte öffnen und konfigurieren für Zuordnung
temp_cameras = []
for cam_id in raw_camera_ids:
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    temp_cameras.append(cap)

# Interaktive Zuordnung
for idx, cap in enumerate(temp_cameras):
    print(f"\n--- Kamera mit Geräte-ID {raw_camera_ids[idx]} ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Fehler beim Lesen von Kamera {raw_camera_ids[idx]}")
            break

        cv2.imshow("Kameraansicht", frame)

        print("Gib die Kameranummer ein (1, 2 oder 3), oder drücke 'w' um das Bild nochmal anzuzeigen:")
        key = cv2.waitKey(0) & 0xFF

        if key in [ord('1'), ord('2'), ord('3')]:
            kamera_nr = int(chr(key))
            if kamera_nr in camera_names:
                print(f"⚠️ Kamera {kamera_nr} wurde bereits zugewiesen!")
                continue
            camera_names[kamera_nr] = raw_camera_ids[idx]
            print(f"Kamera {raw_camera_ids[idx]} wurde als Kamera {kamera_nr} zugeordnet.")
            break
        elif key == ord('q'):
            print("Abbruch durch Benutzer.")
            break
        else:
            print("Ungültige Eingabe.")

    cv2.destroyWindow("Kameraansicht")

# Kameras freigeben
for cap in temp_cameras:
    cap.release()
cv2.destroyAllWindows()

# Ergebnis anzeigen
print("\n✅ Kamerazuweisung abgeschlossen:")
for name, cam_id in camera_names.items():
    print(f"Kamera {name}: Geräte-ID {cam_id}")

# Aufnahme-Einstellungen
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SAVE_PATH = "aufnahmen"
os.makedirs(SAVE_PATH, exist_ok=True)

# Videoaufnahme-Einstellungen
VIDEO_DURATION_SECONDS = 1
FPS = 20
VIDEO_FRAME_COUNT = VIDEO_DURATION_SECONDS * FPS
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')  # oder 'mp4v' für .mp4

# Kameras neu öffnen
sorted_ids = [camera_names[k] for k in sorted(camera_names.keys())]
cameras = []
for cam_id in sorted_ids:
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cameras.append(cap)

previous_frames = [None] * len(cameras)
movement_threshold = 350000

cooldown_frames = 30
cooldown = 0
recording = False

set_counter = 1
pfeil_in_set = 1

print("\nDrücke 's' zum Starten, 'e' zum Stoppen der Aufnahme, 'q' zum Beenden.")

while True:
    current_frames = []
    movement_detected = False

    for idx, cap in enumerate(cameras):
        ret, frame = cap.read()
        if not ret:
            current_frames.append(None)
            continue
        current_frames.append(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if previous_frames[idx] is None:
            previous_frames[idx] = gray
            continue

        delta = cv2.absdiff(previous_frames[idx], gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        movement_score = np.sum(thresh)

        if movement_score > movement_threshold and cooldown == 0:
            movement_detected = True

        previous_frames[idx] = gray

    # Vorschau zusammensetzen
    display = [f if f is not None else np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
               for f in current_frames]
    try:
        combined = cv2.hconcat(display)
    except:
        combined = display[0]

    # Statusanzeige
    cv2.putText(combined, f"Set: {set_counter}  Pfeil: {pfeil_in_set}/3", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    status_text = "AUFNAHME AKTIV" if recording else "PAUSIERT"
    color = (0, 0, 255) if recording else (128, 128, 128)
    cv2.putText(combined, status_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Live-Vorschau", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Programm wird beendet.")
        break
    elif key == ord('s'):
        recording = True
        print("🔴 Aufnahme gestartet.")
    elif key == ord('e'):
        recording = False
        print("⏹️ Aufnahme gestoppt.")

    if movement_detected and recording:
        print(f"🎬 Bewegung erkannt – Aufnahme Pfeil {pfeil_in_set}/3 in Set {set_counter}...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        identifier = f"set_{set_counter:03d}_pfeil{pfeil_in_set}"

        video_writers = []
        filenames = []
        for idx, cam_id in enumerate(sorted_ids):
            filename = os.path.join(SAVE_PATH, f"{identifier}_cam{cam_id}_{timestamp}.avi")
            writer = cv2.VideoWriter(filename, FOURCC, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            video_writers.append(writer)
            filenames.append(filename)

        for _ in range(VIDEO_FRAME_COUNT):
            for idx, cap in enumerate(cameras):
                ret, frame = cap.read()
                if ret:
                    video_writers[idx].write(frame)

            if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q'):
                break

        for writer in video_writers:
            writer.release()

        print("💾 Videos gespeichert:")
        for fname in filenames:
            print(f"   {fname}")

        pfeil_in_set += 1
        cooldown = cooldown_frames

        if pfeil_in_set > 3:
            print(f"✅ Set {set_counter} abgeschlossen.")
            set_counter += 1
            pfeil_in_set = 1

    if cooldown > 0:
        cooldown -= 1

# Aufräumen
for cap in cameras:
    cap.release()
cv2.destroyAllWindows()
