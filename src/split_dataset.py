import os, random, shutil

frames = "data/frames"
labels = "data/annotated"

train_img = "data/train/images"
train_lbl = "data/train/labels"
test_img = "data/test/images"
test_lbl = "data/test/labels"

os.makedirs(train_img, exist_ok=True)
os.makedirs(train_lbl, exist_ok=True)
os.makedirs(test_img, exist_ok=True)
os.makedirs(test_lbl, exist_ok=True)

images = []
for cls in os.listdir(frames):
    for img in os.listdir(f"{frames}/{cls}"):
        images.append(f"{cls}/{img}")

random.shuffle(images)
split = int(0.8 * len(images))

train = images[:split]
test = images[split:]

def move(files, img_dest, lbl_dest):
    for f in files:
        img_src = f"{frames}/{f}"
        name = os.path.splitext(os.path.basename(f))[0]
        lbl_src = f"{labels}/{name}.txt"

        shutil.copy(img_src, img_dest)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dest)

move(train, train_img, train_lbl)
move(test, test_img, test_lbl)

print("Dataset split done")
