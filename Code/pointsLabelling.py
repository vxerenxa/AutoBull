import sys
import os
import re
import json
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QComboBox, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect

from PIL import Image
import numpy as np

# ==== Pfade anpassen ====
DATA_DIR = "./Data/snapshots"
LABEL_DIR = "./Data/labels"
LABELS_JSON = "./Data/points/points_labels.json"
LABELS_CSV = "./Data/points/points_labels.csv"
IMAGE_SIZE = (480, 360)
ZOOM_SIZE = 200

ZONE_OPTIONS = ["Single", "Double", "Triple", "Outer Bull", "Inner Bull", "Miss"]

def parse_yolo_boxes(label_path, img_width, img_height):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, w, h = map(float, parts)
            x1 = int((cx - w/2) * img_width)
            y1 = int((cy - h/2) * img_height)
            x2 = int((cx + w/2) * img_width)
            y2 = int((cy + h/2) * img_height)
            boxes.append((x1, y1, x2, y2))
    return boxes

def collect_image_sets():
    pattern = re.compile(r"(set_\d+)_pfeil([1-3])_cam([0-9])_.*\.jpg")
    sets = {}
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".jpg"):
            continue
        match = pattern.match(file)
        if match:
            set_id, pfeil, cam = match.groups()
            sets.setdefault(set_id, {}).setdefault(f"pfeil{pfeil}", {})[f"cam{cam}"] = file
    return sorted(sets.items())

def load_image(file_path):
    img = Image.open(file_path).resize(IMAGE_SIZE)
    return img

def draw_boxes(image, boxes):
    qimage = image.convert("RGB").copy()
    arr = np.array(qimage)
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    painter = QPainter(pix)
    pen = QPen(QColor(255, 0, 0), 3)
    painter.setPen(pen)
    for x1, y1, x2, y2 in boxes:
        painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
    painter.end()
    return pix

class LabelingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dart Labeling Tool (PyQt5)")
        self.image_sets = collect_image_sets()
        self.set_index = 0
        self.pfeil_index = 0
        self.all_labels = self.load_labels()
        self.zoom_mode = False
        self.zoom_center = None
        self.zoomed_cam = None

        self.init_ui()
        self.update_display()

    def init_ui(self):
        layout = QVBoxLayout()

        # === Bildanzeige: 2 oben, 1 unten ===
        top_row = QHBoxLayout()
        self.image_labels = [QLabel(self) for _ in range(3)]

        for i in range(2):  # cam0, cam2
            self.image_labels[i].setFixedSize(*IMAGE_SIZE)
            self.image_labels[i].mousePressEvent = lambda event, cam=i: self.handle_zoom(event, cam)
            top_row.addWidget(self.image_labels[i])

        layout.addLayout(top_row)

        # cam4 unten zentriert
        bottom_row = QHBoxLayout()
        self.image_labels[2].setFixedSize(*IMAGE_SIZE)
        self.image_labels[2].mousePressEvent = lambda event, cam=2: self.handle_zoom(event, cam)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.image_labels[2])
        bottom_row.addStretch(1)
        layout.addLayout(bottom_row)

        # Eingaben
        self.score_input = QLineEdit(self)
        self.score_input.setPlaceholderText("Gesamtscore")
        self.value_input = QLineEdit(self)
        self.value_input.setPlaceholderText("Pfeilwert")
        self.zone_box = QComboBox()
        self.zone_box.addItems(ZONE_OPTIONS)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.score_input)
        input_layout.addWidget(self.value_input)
        input_layout.addWidget(self.zone_box)
        layout.addLayout(input_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.back_btn = QPushButton("Zurück")
        self.next_btn = QPushButton("Weiter")
        self.unzoom_btn = QPushButton("Zoom zurück")

        self.back_btn.clicked.connect(self.prev_step)
        self.next_btn.clicked.connect(self.next_step)
        self.unzoom_btn.clicked.connect(self.exit_zoom)

        button_layout.addWidget(self.back_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.unzoom_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_labels(self):
        if os.path.exists(LABELS_JSON):
            with open(LABELS_JSON, "r") as f:
                return json.load(f)
        return {}

    def save_labels(self):
        os.makedirs(os.path.dirname(LABELS_JSON), exist_ok=True)
        with open(LABELS_JSON, "w") as f:
            json.dump(self.all_labels, f, indent=2)
        with open(LABELS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            headers = ["set_id"]
            for i in range(1, 4):
                headers += [f"score{i}", f"pfeil{i}_value", f"pfeil{i}_zone"]
            writer.writerow(headers)
            for sid, val in self.all_labels.items():
                row = [sid]
                for i in range(1, 4):
                    row += [val.get(f"score{i}", ""), val.get(f"pfeil{i}_value", ""), val.get(f"pfeil{i}_zone", "")]
                writer.writerow(row)

    def update_display(self):
        if self.set_index >= len(self.image_sets):
            self.close()
            return
        set_id, data = self.image_sets[self.set_index]
        pfeil_key = f"pfeil{self.pfeil_index+1}"

        for i, cam in enumerate(["cam0", "cam2", "cam4"]):
            label = self.image_labels[i]
            filename = data.get(pfeil_key, {}).get(cam)
            if filename:
                img_path = os.path.join(DATA_DIR, filename)
                label_path = os.path.join(LABEL_DIR, filename.replace(".jpg", ".txt"))
                img = load_image(img_path)
                boxes = parse_yolo_boxes(label_path, *IMAGE_SIZE)
                if self.zoom_mode and i == self.zoomed_cam and self.zoom_center:
                    x, y = self.zoom_center
                    x1 = max(0, x - ZOOM_SIZE // 2)
                    y1 = max(0, y - ZOOM_SIZE // 2)
                    x2 = min(IMAGE_SIZE[0], x + ZOOM_SIZE // 2)
                    y2 = min(IMAGE_SIZE[1], y + ZOOM_SIZE // 2)
                    img = img.crop((x1, y1, x2, y2)).resize(IMAGE_SIZE)
                pix = draw_boxes(img, boxes)
                label.setPixmap(pix)
            else:
                label.setText("Kein Bild")

        # Eingabefelder füllen
        current = self.all_labels.get(set_id, {})
        self.score_input.setText(str(current.get(f"score{self.pfeil_index+1}", "")))
        self.value_input.setText(str(current.get(f"pfeil{self.pfeil_index+1}_value", "")))
        zone = current.get(f"pfeil{self.pfeil_index+1}_zone", "Single")
        if zone in ZONE_OPTIONS:
            self.zone_box.setCurrentText(zone)

    def next_step(self):
        set_id, _ = self.image_sets[self.set_index]
        try:
            score = int(self.score_input.text())
            value = int(self.value_input.text())
            zone = self.zone_box.currentText()
        except ValueError:
            print("⚠️ Ungültige Eingaben")
            return
        if set_id not in self.all_labels:
            self.all_labels[set_id] = {}
        self.all_labels[set_id][f"score{self.pfeil_index+1}"] = score
        self.all_labels[set_id][f"pfeil{self.pfeil_index+1}_value"] = value
        self.all_labels[set_id][f"pfeil{self.pfeil_index+1}_zone"] = zone
        self.save_labels()
        self.zoom_mode = False
        self.zoom_center = None
        self.zoomed_cam = None
        if self.pfeil_index < 2:
            self.pfeil_index += 1
        else:
            self.pfeil_index = 0
            self.set_index += 1
        self.update_display()

    def prev_step(self):
        self.zoom_mode = False
        self.zoom_center = None
        self.zoomed_cam = None
        if self.pfeil_index > 0:
            self.pfeil_index -= 1
        elif self.set_index > 0:
            self.set_index -= 1
            self.pfeil_index = 2
        self.update_display()

    def exit_zoom(self):
        self.zoom_mode = False
        self.zoom_center = None
        self.zoomed_cam = None
        self.update_display()

    def handle_zoom(self, event, cam_index):
        x = int(event.pos().x() * IMAGE_SIZE[0] / self.image_labels[cam_index].width())
        y = int(event.pos().y() * IMAGE_SIZE[1] / self.image_labels[cam_index].height())
        self.zoom_mode = True
        self.zoomed_cam = cam_index
        self.zoom_center = (x, y)
        self.update_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelingApp()
    window.show()
    sys.exit(app.exec_())
