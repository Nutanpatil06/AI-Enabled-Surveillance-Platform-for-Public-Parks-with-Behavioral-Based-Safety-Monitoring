import cv2, os

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Can't open:", video_path)
        return

    i = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % 30 == 0:  # ~1 frame/sec
            cv2.imwrite(f"{output_dir}/{os.path.basename(video_path)}_{saved}.jpg", frame)
            saved += 1
        i += 1

    cap.release()
    print(f"✅ {saved} frames from {video_path}")
