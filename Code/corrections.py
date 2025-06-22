import os

image_dir = "./Data/snapshots"
label_dir = "./Data/labels"

for img in os.listdir(image_dir):
    if img.endswith(".jpg"):
        label = img.replace(".jpg", ".txt")
        label_path = os.path.join(label_dir, label)
        if not os.path.exists(label_path):
            open(label_path, "w").close()