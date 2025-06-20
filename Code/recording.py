import cv2
import time
import os

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
VIDEO_FPS = 30
SAVE_PATH = "aufnahmen"

os.makedirs(SAVE_PATH, exist_ok=True)

# Kameras neu öffnen nach Zuordnung (in korrekter Reihenfolge 1, 2, 3)
sorted_ids = [camera_names[k] for k in sorted(camera_names.keys())]
cameras = []
for cam_id in sorted_ids:
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cameras.append(cap)

print("\nDrücke 's' um die Aufnahme zu starten.")
print("Drücke 'e' um die Aufnahme zu beenden.")
print("Drücke 'q' um das Programm zu beenden.")

recording = False
frames_per_camera = [[] for _ in cameras]
clip_counter = 1

while True:
    current_frames = []
    for cap in cameras:
        ret, frame = cap.read()
        current_frames.append(frame if ret else None)

    # Zeige kombinierte Vorschau
    preview = [f if f is not None else cv2.imread("black.jpg") for f in current_frames]
    try:
        combined = cv2.hconcat(preview)
    except:
        combined = preview[0]
    label = "Aufnahme läuft..." if recording else "Warte auf 's' zum Starten"
    cv2.putText(combined, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if recording else (0, 0, 255), 2)
    cv2.imshow("Multi-Kamera Vorschau", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and not recording:
        print(f"🎥 Starte Aufnahme {clip_counter} – werfe deine 3 Pfeile.")
        recording = True
        frames_per_camera = [[] for _ in cameras]

    elif key == ord('e') and recording:
        print("⏹️ Aufnahme wird gespeichert...")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for idx, cam_id in enumerate(sorted_ids):
            video_filename = os.path.join(SAVE_PATH, f"clip_{clip_counter}_cam{cam_id}_{timestamp}.avi")
            if frames_per_camera[idx]:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_filename, fourcc, VIDEO_FPS,
                                      (FRAME_WIDTH, FRAME_HEIGHT))
                for f in frames_per_camera[idx]:
                    out.write(f)
                out.release()
                print(f"💾 Kamera {cam_id} gespeichert: {video_filename}")
            else:
                print(f"⚠️ Keine Frames für Kamera {cam_id} aufgenommen.")

        clip_counter += 1
        recording = False

    elif key == ord('q'):
        print("Programm wird beendet.")
        break

    # Während Aufnahme: Frames speichern
    if recording:
        for idx, frame in enumerate(current_frames):
            if frame is not None:
                frames_per_camera[idx].append(frame.copy())

# Aufräumen
for cap in cameras:
    cap.release()
cv2.destroyAllWindows()
