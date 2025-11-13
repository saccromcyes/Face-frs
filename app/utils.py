
import cv2
import numpy as np

def read_image_bytes(file_bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def crop_box(img, box):
    x1, y1, x2, y2 = box
    h, w = img.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    return img[y1:y2, x1:x2]

# basic resize + normalization for embedder
def preprocess_for_embedding(cropped_face, size=(160,160)):
    face = cv2.resize(cropped_face, size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32') / 255.0
    # mean/std normalization if required by model
    face = (face - 0.5) / 0.5
    # CHW
    face = np.transpose(face, (2,0,1))
    return face
