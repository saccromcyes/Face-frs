import sqlite3
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load models
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load DB
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cursor.execute("SELECT name, embedding FROM faces")
rows = cursor.fetchall()
conn.close()

known_faces = []
for name, emb_blob in rows:
    known_faces.append((name, np.frombuffer(emb_blob, dtype=np.float32)))

# Test recognition
img = Image.open("gallery_images/kinnu_63dd0e1c30ed06f9.jpg")
face = mtcnn(img)
if face is None:
    print("No face detected in test image.")
else:
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0)).numpy().flatten()

    for name, known_emb in known_faces:
        dist = np.linalg.norm(emb - known_emb)
        print(f"Distance from {name}: {dist}")
