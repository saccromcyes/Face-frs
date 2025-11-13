# main.py — FaceFenix Compatible Backend
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch
import numpy as np
import sqlite3
import os
import io
import uvicorn

app = FastAPI(title="FaceFenix API", version="1.0")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database + folders
DB_PATH = "faces.db"
conn = sqlite3.connect(DB_PATH)

os.makedirs("gallery_images", exist_ok=True)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---------- Helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        embedding BLOB,
        image_path TEXT,
        added_on TEXT
    )''')
    conn.commit()
    conn.close()

def get_embedding(image: Image.Image):
    face = mtcnn(image)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected in image.")
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy().flatten()
    return emb

init_db()

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/add_identity")
async def add_identity(name: str = Form(...), file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    embedding = get_embedding(image)

    image_filename = f"{name}_{os.urandom(8).hex()}.jpg"
    image_path = os.path.join("gallery_images", image_filename)
    image.save(image_path)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO faces (name, embedding, image_path, added_on) VALUES (?, ?, ?, datetime('now'))",
        (name, embedding.astype(np.float32).tobytes(), image_path),
    )
    conn.commit()
    conn.close()

    return {"id": cursor.lastrowid, "name": name, "image_path": image_path}

@app.get("/list_identities")
def list_identities():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, image_path, added_on FROM faces")
    records = cursor.fetchall()
    conn.close()

    return [
        {"id": r[0], "name": r[1], "image_path": r[2], "added_on": r[3]}
        for r in records
    ]
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    face = mtcnn(image)

    if face is None:
        return {"match": None, "similarity": 0.0, "distance": 999.0}

    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy().flatten()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM faces")
    records = cursor.fetchall()
    conn.close()

    if not records:
        raise HTTPException(status_code=400, detail="No registered faces found.")

    best_match = None
    best_distance = float("inf")

    for db_name, db_embed in records:
        db_embedding = np.frombuffer(db_embed, dtype=np.float32)
        distance = np.linalg.norm(emb - db_embedding)
        if distance < best_distance:
            best_distance = distance
            best_match = db_name

    similarity = max(0.0, 1.0 - best_distance)

    # ✅ unified response for Streamlit
    if best_distance <= 0.8:
        return {
            "match": best_match,
            "similarity": round(float(similarity), 4),
            "distance": round(float(best_distance), 4)
        }
    else:
        return {
            "match": None,
            "similarity": round(float(similarity), 4),
            "distance": round(float(best_distance), 4)
        }



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
