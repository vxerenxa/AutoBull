import sys
import os
import re
import json
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QFormLayout
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect
from PIL import Image
import numpy as np

# ==== Pfade anpassen ====
DATA_DIR = "./Data/snapshots"
LABEL_DIR = "./Data/bbox_labels"
LABELS_JSON = "./Data/points/points_labels.json"
LABELS_CSV = "./Data/points/points_labels.csv"
IMAGE_SIZE = (480, 360)
ZOOM_SIZE = 200

SCORE_OPTIONS = [str(i) for i in range(1, 21)] + ["Bull", "Miss"]
FESTE_FARBEN = ["Rot", "Grün", "Blau"]
FARBEN_RGB = {
    "Rot": (255, 0, 0),
    "Grün": (0, 255, 0),
    "Blau": (0, 0, 255)
}

# Hilfsstruktur zur globalen Box-ID-Zuordnung pro Set
BOX_CACHE = {}


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


def get_consistent_box_order(set_id, boxes):
    global BOX_CACHE
    if set_id not in BOX_CACHE:
        BOX_CACHE[set_id] = boxes[:3]  # Save first occurrence
        return boxes[:3]
    ref_boxes = BOX_CACHE[set_id]
    matched = []
    used = set()
    for ref in ref_boxes:
        best = None
        best_dist = float("inf")
        for i, b in enumerate(boxes):
            if i in used:
                continue
            dist = (b[0] - ref[0]) ** 2 + (b[1] - ref[1]) ** 2
            if dist < best_dist:
                best = b
                best_dist = dist
                best_idx = i
        if best is not None:
            matched.append(best)
            used.add(best_idx)
    return matched


def collect_image_sets():
    sets = {}
    pattern = re.compile(r'(set_\d+)_pfeil\d+_cam(\d)_.*\.jpg')
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".jpg"):
            continue
        match = pattern.match(file)
        if match:
            set_id, cam = match.groups()
            sets.setdefault(set_id, {})[f"cam{cam}"] = file
    return sorted(sets.items())


# ==== PATCH: apply consistent box order in display ====
def apply_box_order_to_image(img_path, label_path, set_id):
    img = load_image(img_path)
    boxes = parse_yolo_boxes(label_path, *IMAGE_SIZE)
    boxes = get_consistent_box_order(set_id, boxes)
    return img, boxes


def load_image(file_path):
    img = Image.open(file_path).resize(IMAGE_SIZE)
    return img



def draw_boxes(image, boxes, labels=None, colors=None):
    qimage = image.convert("RGB").copy()
    arr = np.array(qimage)
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    painter = QPainter(pix)

    if colors is None:
        colors = [(255, 0, 0)] * len(boxes)
    if labels is None:
        labels = [""] * len(boxes)

    for (x1, y1, x2, y2), color, label_text in zip(boxes, colors, labels):
        pen = QPen(QColor(*color), 3)
        painter.setPen(pen)
        painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
        painter.drawText(x1 + 5, y1 + 20, label_text)

    painter.end()
    return pix


class LabelingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dart Labeling Tool - Nur Punkte + feste Farben")
        self.image_sets = collect_image_sets()
        self.all_labels = self.load_labels()
        self.set_index = self.find_last_labeled_index()

        self.zoom_mode = False
        self.zoom_center = None
        self.zoomed_cam = None

        self.init_ui()
        self.update_display()

    def find_last_labeled_index(self):
        if not self.all_labels:
            return 0
        last_labeled_set = list(self.all_labels.keys())[-1]
        for idx, (set_id, _) in enumerate(self.image_sets):
            if set_id == last_labeled_set:
                return min(idx + 1, len(self.image_sets) - 1)
        return 0

    def init_ui(self):
        layout = QVBoxLayout()
        self.image_labels = [QLabel(self) for _ in range(3)]
        for i, label in enumerate(self.image_labels):
            label.setFixedSize(*IMAGE_SIZE)
            label.mousePressEvent = self.make_zoom_handler(i)
        layout.addLayout(self.build_image_layout())

        self.score_boxes = []
        self.input_layout = QFormLayout()
        layout.addLayout(self.input_layout)

        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("Speichern & Weiter")
        self.back_btn = QPushButton("Zurück")
        self.unzoom_btn = QPushButton("Zoom zurück")

        self.save_btn.clicked.connect(self.save_and_next)
        self.back_btn.clicked.connect(self.prev_step)
        self.unzoom_btn.clicked.connect(self.reset_zoom)

        button_layout.addWidget(self.back_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.unzoom_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def make_zoom_handler(self, cam_idx):
        def handler(event):
            x = int(event.pos().x() * IMAGE_SIZE[0] / self.image_labels[cam_idx].width())
            y = int(event.pos().y() * IMAGE_SIZE[1] / self.image_labels[cam_idx].height())
            self.zoom_mode = True
            self.zoomed_cam = cam_idx
            self.zoom_center = (x, y)
            self.update_display()
        return handler

    def build_image_layout(self):
        layout = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(self.image_labels[0])
        top.addWidget(self.image_labels[1])
        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(self.image_labels[2])
        bottom.addStretch()
        layout.addLayout(top)
        layout.addLayout(bottom)
        return layout

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
            writer.writerow(["set_id", "wurf1", "farbe1", "wurf2", "farbe2", "wurf3", "farbe3", "total_score"])
            for sid, val in self.all_labels.items():
                row = [
                    sid,
                    val.get("wurf1", ""), val.get("farbe1", ""),
                    val.get("wurf2", ""), val.get("farbe2", ""),
                    val.get("wurf3", ""), val.get("farbe3", ""),
                    val.get("total_score", "")
                ]
                writer.writerow(row)

    def update_display(self):
        if self.set_index >= len(self.image_sets):
            self.close()
            return

        set_id, cam_files = self.image_sets[self.set_index]

        for i, cam in enumerate(sorted(cam_files.keys())):
            if i >= len(self.image_labels):
                continue
            label = self.image_labels[i]
            filename = cam_files[cam]
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

                zoomed_boxes = []
                scale_x = IMAGE_SIZE[0] / (x2 - x1)
                scale_y = IMAGE_SIZE[1] / (y2 - y1)
                for bx1, by1, bx2, by2 in boxes:
                    if bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2:
                        continue
                    zx1 = int((bx1 - x1) * scale_x)
                    zy1 = int((by1 - y1) * scale_y)
                    zx2 = int((bx2 - x1) * scale_x)
                    zy2 = int((by2 - y1) * scale_y)
                    zoomed_boxes.append((zx1, zy1, zx2, zy2))
                boxes = zoomed_boxes

            farben = list(FARBEN_RGB.values())
            labels = [f"Wurf {i+1}" for i in range(len(boxes))]
            pix = draw_boxes(img, boxes, labels=labels, colors=farben[:len(boxes)])
            label.setPixmap(pix)

        while self.input_layout.count():
            item = self.input_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.score_boxes = []

        current = self.all_labels.get(set_id, {})

        prev_values = {}
        if self.set_index > 0:
            prev_id, _ = self.image_sets[self.set_index - 1]
            prev_values = self.all_labels.get(prev_id, {})

        for i in range(3):
            score_box = QComboBox()
            score_box.addItems(SCORE_OPTIONS)
            default_score = current.get(f"wurf{i+1}", prev_values.get(f"wurf{i+1}", SCORE_OPTIONS[0]))
            score_box.setCurrentText(default_score)
            self.score_boxes.append(score_box)
            farb_label = QLabel(FESTE_FARBEN[i])
            hbox = QHBoxLayout()
            hbox.addWidget(score_box)
            hbox.addWidget(farb_label)
            self.input_layout.addRow(f"Wurf {i+1}:", hbox)

    def save_and_next(self):
        if self.set_index >= len(self.image_sets):
            return
        set_id, _ = self.image_sets[self.set_index]
        self.all_labels.setdefault(set_id, {})

        total = 0
        for i in range(len(self.score_boxes)):
            val = self.score_boxes[i].currentText()
            farbe = FESTE_FARBEN[i]
            self.all_labels[set_id][f"wurf{i+1}"] = val
            self.all_labels[set_id][f"farbe{i+1}"] = farbe
            if val.isdigit():
                total += int(val)
            elif val == "Bull":
                total += 50

        self.all_labels[set_id]["total_score"] = total
        self.save_labels()
        self.set_index += 1
        self.update_display()

    def prev_step(self):
        if self.set_index > 0:
            self.set_index -= 1
            self.update_display()

    def reset_zoom(self):
        self.zoom_mode = False
        self.zoom_center = None
        self.zoomed_cam = None
        self.update_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelingApp()
    window.show()
    sys.exit(app.exec_())
