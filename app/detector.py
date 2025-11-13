from app.config import DEVICE, DETECTOR_BACKEND
import numpy as np

# Option A: MTCNN via facenet-pytorch (good for prototype)
try:
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None

# Option B: insightface retinaface detector (production)
try:
    import insightface
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

class Detector:
    def __init__(self, backend="mtcnn"):
        self.backend = backend
        if backend == "mtcnn":
            if MTCNN is None:
                raise RuntimeError("facenet-pytorch not installed. pip install facenet-pytorch")
            self.det = MTCNN(keep_all=True, device=DEVICE)
        elif backend == "insightface":
            if FaceAnalysis is None:
                raise RuntimeError("insightface not installed. pip install insightface")
            self.det = FaceAnalysis(allowed_modules=['detection'], providers=['CPUExecutionProvider'])
            self.det.prepare(ctx_id=0, det_thresh=0.5)
        else:
            raise ValueError("Unknown backend")

    def detect(self, img_bgr):
        # img_bgr: numpy BGR img
        if self.backend == "mtcnn":
            boxes, probs = self.det.detect(img_bgr[..., ::-1])  # MTCNN expects RGB
            if boxes is None:
                return []
            out = []
            for b, p in zip(boxes, probs):
                x1,y1,x2,y2 = map(float, b)
                out.append({"box": [x1,y1,x2,y2], "score": float(p)})
            return out
        else:
            faces = self.det.get(img_bgr[..., ::-1])  # insightface expects RGB
            out = []
            for f in faces:
                x1,y1,x2,y2 = f.bbox.astype(float)
                out.append({"box": [float(x1),float(y1),float(x2),float(y2)], "score": float(f.det_score)})
            return out
