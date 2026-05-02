import streamlit as st
import cv2
import os
import time
import pandas as pd
import hashlib
from ultralytics import YOLO
from tempfile import NamedTemporaryFile
from collections import deque

# ================= CONFIG =================
st.set_page_config(page_title="AI Park Surveillance", layout="wide")

# ================= DARK UI – TEXT VISIBILITY FIX =================
st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #0b1220;
}

/* Force readable text everywhere */
html, body, [class*="css"], p, span, label, div {
    color: #e5e7eb !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Buttons */
.stButton > button {
    background-color: #1e293b !important;
    color: #ffffff !important;
    border-radius: 12px;
    border: none;
    font-weight: 600;
}

/* Inputs */
input, textarea {
    background-color: #020617 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #020617 !important;
    border-radius: 12px;
}
[data-testid="stFileUploader"] * {
    color: #ffffff !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #ffffff !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #020617;
    border-radius: 14px;
    padding: 14px;
}
[data-testid="stMetric"] * {
    color: #ffffff !important;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    background-color: #020617;
    border-radius: 12px;
}
[data-testid="stDataFrame"] * {
    color: #ffffff !important;
}

/* Alerts */
.stAlert, .stAlert * {
    color: #ffffff !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background-color: #38bdf8 !important;
}

</style>
""", unsafe_allow_html=True)

UPLOAD_DIR = "backend/uploads"
PROCESSED_DIR = "backend/processed"
MODEL_PATH = "yolov8n.pt"
USERS_FILE = "users.csv"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

UNAUTHORIZED_CLASSES = ["car", "motorcycle", "bicycle", "bus", "truck"]

CONF_THRESHOLDS = {
    "person": 0.25,
    "bicycle": 0.30,
    "motorcycle": 0.30,
    "car": 0.35,
    "bus": 0.35,
    "truck": 0.35
}

# ================= HELPERS =================
def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE)
    return pd.DataFrame(columns=["name", "email", "password"])

def save_user(n, e, p):
    df = load_users()
    df.loc[len(df)] = [n, e, hash_password(p)]
    df.to_csv(USERS_FILE, index=False)

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def enhance_frame(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

# ================= SESSION =================
for k in ["logged_in", "play_video", "video_processed", "processed_path", "last_video_name"]:
    if k not in st.session_state:
        st.session_state[k] = False if "in" in k else None

# ================= AUTH =================
def auth_ui():
    st.title("🔐 User Authentication")
    login, register = st.tabs(["Login", "Register"])

    with login:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            users = load_users()
            if not users[
                (users.email == email) &
                (users.password == hash_password(password))
            ].empty:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    with register:
        name = st.text_input("Name")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_pass")
        confirm = st.text_input("Confirm Password", type="password")
        if st.button("Register"):
            save_user(name, email, password)
            st.success("Registered successfully")

if not st.session_state.logged_in:
    auth_ui()
    st.stop()

def logout():
    st.session_state.logged_in = False
    st.rerun()

# ================= MODEL =================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ================= DASHBOARD =================
st.title("🌳 AI-Based Park Surveillance System")
st.button("Logout", on_click=logout)

video_file = st.file_uploader("Upload Park Video", type=["mp4", "avi"])
progress_bar = st.progress(0)
progress_text = st.empty()

# ================= RESET =================
if video_file and video_file.name != st.session_state.last_video_name:
    st.session_state.video_processed = False
    st.session_state.last_video_name = video_file.name

# ================= PROCESS VIDEO =================
if video_file and not st.session_state.video_processed:
    tmp = NamedTemporaryFile(delete=False)
    tmp.write(video_file.read())

    output_path = os.path.join(PROCESSED_DIR, f"processed_{int(time.time())}.mp4")
    cap = cv2.VideoCapture(tmp.name)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 720)
    )

    authorized = unauthorized = 0
    flagged_logs = []
    frame_no = 0
    recent_unauth = deque(maxlen=3)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = enhance_frame(cv2.resize(frame, (1280, 720)))
        results = model(frame, conf=0.25, verbose=False)

        persons, vehicles = [], []

        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLDS.get(label, 0.25):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) * (y2 - y1) < 800:
                continue

            if label == "person":
                persons.append((x1, y1, x2, y2, conf))
            elif label in UNAUTHORIZED_CLASSES:
                vehicles.append((x1, y1, x2, y2, label, conf))

        used = set()
        frame_unauth = 0

        for vx1, vy1, vx2, vy2, vlabel, vconf in vehicles:
            merged = False
            for i, (px1, py1, px2, py2, pconf) in enumerate(persons):
                if iou((vx1, vy1, vx2, vy2), (px1, py1, px2, py2)) > 0.25 and pconf > 0.35:
                    merged = True
                    used.add(i)
                    frame_unauth += 1

                    x1, y1 = min(vx1, px1), min(vy1, py1)
                    x2, y2 = max(vx2, px2), max(vy2, py2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(frame, f"unauthorized {vconf:.2f}",
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0,0,255), 2)

                    flagged_logs.append({
                        "Time (sec)": round(frame_no / fps, 2),
                        "Activity": f"person + {vlabel}",
                        "Status": "RED",
                        "Confidence": round(vconf, 2)
                    })
                    break

            if not merged:
                frame_unauth += 1
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0,0,255), 2)
                cv2.putText(frame, f"{vlabel} {vconf:.2f}",
                            (vx1, vy1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,0,255), 2)

        for i, (px1, py1, px2, py2, pconf) in enumerate(persons):
            if i not in used:
                authorized += 1
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0,255,0), 2)
                cv2.putText(frame, f"person {pconf:.2f}",
                            (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

        recent_unauth.append(frame_unauth)
        unauthorized += max(recent_unauth)

        out.write(frame)
        frame_no += 1
        pct = int((frame_no / total) * 100) if total else 0
        progress_bar.progress(min(pct, 100))
        progress_text.info(f"Processing: {pct}%")

    cap.release()
    out.release()
    progress_text.success("Processing completed")

    st.session_state.video_processed = True
    st.session_state.processed_path = output_path
    st.session_state.authorized = authorized
    st.session_state.unauthorized = unauthorized
    st.session_state.flagged_logs = flagged_logs

# ================= RESULTS =================
if st.session_state.video_processed:
    st.header("2️⃣ Processed Video Output")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Play"):
            st.session_state.play_video = True
    with col2:
        if st.button("⏹ Stop"):
            st.session_state.play_video = False

    placeholder = st.empty()
    if st.session_state.play_video:
        cap = cv2.VideoCapture(st.session_state.processed_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        while cap.isOpened() and st.session_state.play_video:
            ret, frame = cap.read()
            if not ret:
                break
            placeholder.image(frame, channels="BGR", use_container_width=True)
            time.sleep(1 / fps)
        cap.release()

    st.header("3️⃣ Activity Analytics")
    st.metric("Authorized", st.session_state.authorized)
    st.metric("Unauthorized", st.session_state.unauthorized)

    st.header("4️⃣ Admin Review Panel")
    if st.session_state.flagged_logs:
        st.dataframe(pd.DataFrame(st.session_state.flagged_logs))
    else:
        st.info("No unauthorized activity detected")

    st.header("5️⃣ Download Excel Report")

    summary_df = pd.DataFrame([{
        "Authorized": st.session_state.authorized,
        "Unauthorized": st.session_state.unauthorized,
        "Processed Video": os.path.basename(st.session_state.processed_path)
    }])

    report_path = "park_surveillance_report.xlsx"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if st.session_state.flagged_logs:
            pd.DataFrame(st.session_state.flagged_logs).to_excel(
                writer, sheet_name="Flagged_Activities", index=False
            )

    with open(report_path, "rb") as f:
        st.download_button(
            "📥 Download Excel Report",
            f,
            file_name="park_surveillance_report.xlsx"
        )
