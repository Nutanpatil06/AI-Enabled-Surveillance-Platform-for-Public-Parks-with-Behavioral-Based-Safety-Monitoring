import os
import random

IMAGE_DIR = "data/images/train"
LABEL_DIR = "data/labels/train"

TARGET_WALKING = 500   # keep only 500 walking+standing samples

walking_files = []

# collect walking samples
for label_file in os.listdir(LABEL_DIR):
    path = os.path.join(LABEL_DIR, label_file)
    with open(path, "r") as f:
        cls = f.readline().split()[0]
        if cls == "0":  # walking + standing
            walking_files.append(label_file.replace(".txt", ""))

print("Total walking samples:", len(walking_files))

# remove extra walking samples
if len(walking_files) > TARGET_WALKING:
    remove_count = len(walking_files) - TARGET_WALKING
    remove_files = random.sample(walking_files, remove_count)

    for name in remove_files:
        img = os.path.join(IMAGE_DIR, name + ".jpg")
        label = os.path.join(LABEL_DIR, name + ".txt")

        if os.path.exists(img):
            os.remove(img)
        if os.path.exists(label):
            os.remove(label)

    print("Walking samples reduced to", TARGET_WALKING)

print("Dataset balancing complete.")
