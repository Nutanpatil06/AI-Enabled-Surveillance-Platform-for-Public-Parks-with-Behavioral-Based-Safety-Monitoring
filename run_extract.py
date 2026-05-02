from src.preprocess import extract_frames
import os

for cls in ["walking", "running", "unauthorized"]:
    for vid in os.listdir(f"data/raw_videos/{cls}"):
        extract_frames(f"data/raw_videos/{cls}/{vid}", f"data/frames/{cls}")
