# compare.py
import numpy as np
import sqlite3
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torch

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, device=device)

# Load DB face
conn = sqlite3.connect("faces.db")
cursor = conn.cursor()
cursor.execute("SELECT name, embedding FROM faces LIMIT 1")
row = cursor.fetchone()
conn.close()

name, db_embed = row
db_embedding = np.frombuffer(db_embed, dtype=np.float32)

# New image to compare
image = image = Image.open(r"C:\Users\ndsha\Desktop\face_frs\gallery_images\kinnu_8fc39bab00a871c6.jpg").convert("RGB")
face = mtcnn(image)
emb = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy().flatten()

# Compute cosine similarity
sim = np.dot(emb, db_embedding) / (np.linalg.norm(emb) * np.linalg.norm(db_embedding))
print(f"Similarity with {name}: {sim:.4f}")
