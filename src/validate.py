from collections import Counter
import os

label_path = "data/train/labels"
counter = Counter()

for file in os.listdir(label_path):
    with open(f"{label_path}/{file}") as f:
        for line in f:
            cls = int(line.split()[0])
            counter[cls] += 1

print("Class distribution in training set:")
print(counter)
