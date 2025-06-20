import cv2

# Kamera-IDs
camera_ids = [0, 2, 4]
camera_names = {}

# Kameraobjekte öffnen und konfigurieren
cameras = []
for cam_id in camera_ids:
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cameras.append(cap)

# Für jede Kamera nacheinander Frames anzeigen und Benutzerabfrage durchführen
for idx, cap in enumerate(cameras):
    print(f"\n--- Kamera {camera_ids[idx]} ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Fehler beim Lesen von Kamera {camera_ids[idx]}")
            break

        cv2.imshow("Kameraansicht", frame)

        print("Gib die Kameranummer ein (1, 2 oder 3), oder drücke 'w' um das Bild nochmal anzuzeigen:")
        key = cv2.waitKey(0) & 0xFF

        if key in [ord('1'), ord('2'), ord('3')]:
            camera_names[int(chr(key))] = camera_ids[idx]
            print(f"Kamera {camera_ids[idx]} wurde als Kamera {chr(key)} zugeordnet.")
            break
        elif key == ord('q'):
            print("Abbruch durch Benutzer.")
            break
        else:
            print("Ungültige Eingabe.")

    cv2.destroyWindow("Kameraansicht")

# Kameras freigeben
for cap in cameras:
    cap.release()
cv2.destroyAllWindows()

# Ergebnis anzeigen
print("\nKamerazuweisung:")
for name, cam_id in camera_names.items():
    print(f"Kamera {name}: Geräte-ID {cam_id}")
