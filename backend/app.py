from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles

import os
import shutil
import uuid
import cv2
import pandas as pd
from ultralytics import YOLO

from backend.database import get_db
from backend.auth_utils import hash_password, verify_password

# ================== PATH SETUP (CRITICAL) ==================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== APP INIT ==================

app = FastAPI()

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- SESSION ----
app.add_middleware(
    SessionMiddleware,
    secret_key="super-secret-key-change-this"
)

# ---- STATIC FILES (THIS FIXES JS ISSUE) ----
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ================== GLOBALS ==================

model = None
MODEL_PATH = "yolov8n.pt"

progress_status = {}
analytics_data = {}
flagged_logs = {}

AUTHORIZED_CLASS = "person"
UNAUTHORIZED_CLASSES = ["bicycle", "car", "motorcycle", "bus", "truck"]

CLASS_COLORS = {
    "person": (0, 255, 0),
    "bicycle": (0, 0, 255),
    "car": (0, 0, 255),
    "motorcycle": (0, 0, 255),
    "bus": (0, 0, 255),
    "truck": (0, 0, 255)
}

# ================== HELPERS ==================

def require_login(request: Request):
    return "user" in request.session

# ================== UI ROUTES ==================

@app.get("/", response_class=HTMLResponse)
def login_page():
    with open(os.path.join(FRONTEND_DIR, "login.html"), encoding="utf-8") as f:
        return f.read()

@app.get("/register-ui", response_class=HTMLResponse)
def register_page():
    with open(os.path.join(FRONTEND_DIR, "register.html"), encoding="utf-8") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    if not require_login(request):
        return "<h3>Unauthorized</h3>"
    with open(os.path.join(FRONTEND_DIR, "dashboard.html"), encoding="utf-8") as f:
        return f.read()

# ================== AUTH APIs ==================

@app.post("/register")
def register_user(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    db = get_db()

    existing = db.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()

    if existing:
        db.close()
        return {"error": "Email already registered"}

    db.execute(
        "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
        (name, email, hash_password(password))
    )
    db.commit()
    db.close()

    return {"message": "User registered successfully"}

@app.post("/login")
def login_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()
    db.close()

    if not user or not verify_password(password, user["password"]):
        return {"error": "Invalid email or password"}

    request.session["user"] = {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"]
    }

    return {"message": "Login successful"}

@app.get("/logout")
def logout_user(request: Request):
    request.session.clear()
    return {"message": "Logged out"}

# ================== VIDEO UPLOAD ==================

@app.post("/upload-video")
async def upload_video(
    request: Request,
    video: UploadFile = File(...)
):
    if not require_login(request):
        return {"error": "Unauthorized"}

    global model
    if model is None:
        model = YOLO(MODEL_PATH)

    video_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}_out.mp4")

    progress_status[video_id] = 0
    analytics_data[video_id] = {"authorized": 0, "unauthorized": 0}
    flagged_logs[video_id] = []

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed += 1
        progress_status[video_id] = int((processed / total_frames) * 100)

        results = model(frame, conf=0.4)

        for box in results[0].boxes:
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            if cls == AUTHORIZED_CLASS:
                analytics_data[video_id]["authorized"] += 1
            elif cls in UNAUTHORIZED_CLASSES:
                analytics_data[video_id]["unauthorized"] += 1
                flagged_logs[video_id].append({
                    "time": f"{round(processed / fps, 2)}s",
                    "activity": cls,
                    "status": "RED",
                    "confidence": round(conf, 2)
                })

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        out.write(frame)

    cap.release()
    out.release()
    progress_status[video_id] = 100

    return {"video_id": video_id}

# ================== APIs ==================

@app.get("/progress/{video_id}")
def get_progress(video_id: str):
    return {"progress": progress_status.get(video_id, 0)}

@app.get("/analytics/{video_id}")
def get_analytics(video_id: str):
    return analytics_data.get(video_id, {})

@app.get("/admin/flagged/{video_id}")
def get_logs(video_id: str):
    return flagged_logs.get(video_id, [])

@app.get("/report/{video_id}")
def download_report(video_id: str):
    data = analytics_data.get(video_id)
    logs = flagged_logs.get(video_id, [])

    if not data:
        return {"error": "Invalid video_id"}

    df1 = pd.DataFrame([
        ["Authorized", data["authorized"]],
        ["Unauthorized", data["unauthorized"]]
    ], columns=["Metric", "Value"])

    df2 = pd.DataFrame(logs)

    report_path = os.path.join(OUTPUT_DIR, f"{video_id}_report.xlsx")
    with pd.ExcelWriter(report_path) as writer:
        df1.to_excel(writer, sheet_name="Summary", index=False)
        df2.to_excel(writer, sheet_name="Logs", index=False)

    return FileResponse(report_path, filename=f"{video_id}_report.xlsx")
