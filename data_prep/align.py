# pip install face-alignment
import face_alignment
import cv2
import numpy as np
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

def align_and_crop(img_bgr, output_size=(160,160)):
    # face_alignment expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(img_rgb)
    if preds is None or len(preds)==0:
        return None
    lm = preds[0]  # (68,2) or (5,2) depending
    # compute transformation using eye centers and nose; use similarity transform
    # For brevity you can use a standardized alignment warp (open-source code available)
    # Save cropped aligned face
    # ...
