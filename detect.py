from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/train5/weights/best.pt")

video_path = r"C:\Users\Harsh\Desktop\park_surveillance\data\raw_videos\test4_video.mp4"

print("File exists:", os.path.exists(video_path))

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ ERROR: Video not opened")
    exit()

print("✅ Video opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Park Surveillance - Video Test", annotated)
    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
