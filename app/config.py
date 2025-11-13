import os

DB_PATH = os.environ.get("FACE_DB", "face_gallery.db")
EMBED_DIM = 512
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", 0.45))  # cosine distance (lower = closer)
TOP_K = int(os.environ.get("TOP_K", 5))
DETECTOR_BACKEND = os.environ.get("DETECTOR_BACKEND", "mtcnn")  # options: mtcnn | insightface
EMBEDDER_BACKEND = os.environ.get("EMBEDDER_BACKEND", "facenet")  # facenet | insightface
DEVICE = "cpu"
GALLERY_DIR = os.environ.get("GALLERY_DIR", "gallery_images")
