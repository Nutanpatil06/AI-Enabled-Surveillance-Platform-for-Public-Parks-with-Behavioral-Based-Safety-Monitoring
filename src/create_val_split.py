import os
import random
import shutil

TRAIN_IMG = "data/images/train"
TRAIN_LBL = "data/labels/train"
VAL_IMG = "data/images/val"
VAL_LBL = "data/labels/val"

os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)

images = [f for f in os.listdir(TRAIN_IMG) if f.endswith((".jpg", ".png"))]

val_count = int(0.1 * len(images))  # 10% for validation
val_images = random.sample(images, val_count)

for img in val_images:
    name = os.path.splitext(img)[0]

    # move image
    shutil.move(os.path.join(TRAIN_IMG, img), os.path.join(VAL_IMG, img))

    # move label
    lbl = name + ".txt"
    shutil.move(os.path.join(TRAIN_LBL, lbl), os.path.join(VAL_LBL, lbl))

print(f"Validation set created: {val_count} images")
