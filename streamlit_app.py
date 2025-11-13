# streamlit_app.py — FaceFenix Full Dashboard (Final Version)
# Supports: Health Check · Add Identity · List Identities · Recognize
# Compatible with your FastAPI backend main.py
import os
import streamlit as st
import requests, io, base64
from PIL import Image
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="FaceFenix — Dashboard", layout="wide", initial_sidebar_state="expanded")

# ----------------- CSS / STYLING -----------------
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 10% 10%, #051025, #071426 30%, #02040a 70%); color: #e8f6ff; }
.glass { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:14px; padding:16px; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 10px 30px rgba(2,8,23,0.6); }
.title { font-size:26px; font-weight:700; color:#e6f7ff; }
.muted { color:#9fb8d8; font-size:13px; }
.timeline { max-height: 380px; overflow:auto; padding-right:8px; }
.log-entry { padding:10px; border-radius:10px; margin-bottom:8px; background:linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-left:4px solid rgba(255,255,255,0.04); }
.small { font-size:12px; color:#9fb8d8; }
</style>
""", unsafe_allow_html=True)

# ----------------- Helpers -----------------
def get_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}

def post_add_identity(name: str, img_bytes: bytes):
    files = {"file": ("image.png", img_bytes, "image/png")}
    data = {"name": name}
    try:
        return requests.post(f"{API_BASE}/add_identity", files=files, data=data, timeout=20)
    except Exception:
        return None

def get_list_identities():
    try:
        r = requests.get(f"{API_BASE}/list_identities", timeout=20)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def post_recognize(img_bytes: bytes):
    files = {"file": ("image.png", img_bytes, "image/png")}
    try:
        return requests.post(f"{API_BASE}/recognize", files=files, timeout=20)
    except Exception:
        return None

def pil_from_bytes(b):
    return Image.open(io.BytesIO(b)).convert("RGB")

def make_thumb_bytes(pil_img, size=(96, 96)):
    pil = pil_img.copy()
    pil.thumbnail(size)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return buf.getvalue()

def encode_b64(b):
    return base64.b64encode(b).decode("utf-8")

if "security_log" not in st.session_state:
    st.session_state.security_log = []

# ----------------- Layout -----------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
c1, c2 = st.columns([1.3, 0.9])

with c1:
    st.markdown("<div class='title'>👁️ FaceFenix · Recognition Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Health check, register identities, view list, and recognize faces.</div>", unsafe_allow_html=True)
    st.markdown("---")

    mode = st.radio("Mode", ("Health", "Add Identity", "List Identities", "Recognize"))

    # ---------------- HEALTH MODE ----------------
    if mode == "Health":
        st.markdown("### 🩺 System Health")
        data = get_health()
        if "error" in data:
            st.error(f"❌ {data['error']}")
        else:
            st.success("✅ API is running fine.")
            st.json(data)

    # ---------------- ADD IDENTITY ----------------
    elif mode == "Add Identity":
        st.markdown("### 🧍 Register New Identity")
        uploaded = st.file_uploader("Upload a face image (JPG/PNG):", type=["jpg", "jpeg", "png"])
        cam = st.camera_input("Or capture from webcam")
        name = st.text_input("Enter person's name")
        img_bytes = cam.getvalue() if cam else (uploaded.read() if uploaded else None)

        if img_bytes:
            st.image(img_bytes, caption="Preview", use_container_width=True)

        if st.button("Register Identity"):
            if not img_bytes:
                st.error("Please upload or capture a face image.")
            elif not name.strip():
                st.error("Please enter a valid name.")
            else:
                with st.spinner("Registering face..."):
                    resp = post_add_identity(name.strip(), img_bytes)
                    if resp is None:
                        st.error("❌ Backend not reachable. Is FastAPI running?")
                    elif resp.status_code == 200:
                        st.success(f"✅ Successfully registered {name}")
                        st.balloons()
                    else:
                        st.error(resp.text)

    # ---------------- LIST IDENTITIES ----------------
    elif mode == "List Identities":
        st.markdown("### 📋 Registered Faces in Database")
        records = get_list_identities()
        if not records:
            st.info("No identities registered yet.")
        else:
            for rec in records:
                with st.container():
                    st.markdown(f"**{rec['name']}** — Added on: {rec['added_on']}")
                    if os.path.exists(rec["image_path"]):
                        st.image(rec["image_path"], width=150)
                    else:
                        st.text(f"Image: {rec['image_path']}")

    # ---------------- RECOGNIZE ----------------
    elif mode == "Recognize":
        st.markdown("### 🔍 Face Recognition")
        uploaded = st.file_uploader("Upload a face image (JPG/PNG):", type=["jpg", "jpeg", "png"])
        cam = st.camera_input("Or capture from webcam")
        img_bytes = cam.getvalue() if cam else (uploaded.read() if uploaded else None)

        if img_bytes:
            st.image(img_bytes, caption="Preview", use_container_width=True)

        if st.button("Run Recognition"):
            if not img_bytes:
                st.error("Please upload or capture an image first.")
            else:
                with st.spinner("Analyzing face..."):
                    resp = post_recognize(img_bytes)
                    if resp is None:
                        st.error("❌ Backend not reachable.")
                    elif resp.status_code != 200:
                        st.error(resp.text)
                    else:
                        data = resp.json()
                        match = data.get("match")
                        similarity = data.get("similarity")
                        distance = data.get("distance")

                        if match:
                            score = similarity if similarity is not None else (1 - distance if distance else 0)
                            st.success(f"✅ Match Found: {match} (Confidence: {score*100:.1f}%)")

                            pil = pil_from_bytes(img_bytes)
                            buf = io.BytesIO()
                            pil.save(buf, format="PNG")
                            st.image(buf.getvalue(), caption="Recognition Result", use_container_width=True)

                            thumb_b64 = encode_b64(make_thumb_bytes(pil))
                            st.session_state.security_log.insert(0, {
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "name": match,
                                "score": score,
                                "thumb": thumb_b64,
                            })
                        else:
                            st.warning("❌ No match found.")

# ---------------- SECURITY TIMELINE ----------------
with c2:
    st.markdown("## 🕒 Security Timeline")
    st.markdown("Recent recognitions (latest first).")
    st.markdown("<div class='timeline'>", unsafe_allow_html=True)

    if st.session_state.security_log:
        for e in st.session_state.security_log[:50]:
            name = e.get("name") or "Unknown"
            score = e.get("score")
            thumb_b64 = e.get("thumb")
            t = e.get("time")
            st.markdown("<div class='log-entry'>", unsafe_allow_html=True)
            cols = st.columns([1, 3, 1])
            with cols[0]:
                if thumb_b64:
                    st.image(base64.b64decode(thumb_b64), width=72)
            with cols[1]:
                st.markdown(f"**{name}**")
                st.markdown(f"<div class='small'>{t}</div>", unsafe_allow_html=True)
            with cols[2]:
                if score is not None:
                    pct = float(score) * 100
                    color = "#2be6a0" if score >= 0.9 else "#ffd36b" if score >= 0.75 else "#ff6b6b"
                    st.markdown(
                        f"<div style='text-align:right; font-weight:700; color:{color}'>{pct:.1f}%</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div class='small'>No score</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No recognition events yet.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
