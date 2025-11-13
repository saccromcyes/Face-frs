import numpy as np
from app.config import DEVICE, EMBEDDER_BACKEND
# Option A: facenet-pytorch InceptionResnetV1
try:
    from facenet_pytorch import InceptionResnetV1
except Exception:
    InceptionResnetV1 = None
# Option B: insightface model (ArcFace)
try:
    import insightface
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

class Embedder:
    def __init__(self, backend="facenet"):
        self.backend = backend
        if backend == "facenet":
            if InceptionResnetV1 is None:
                raise RuntimeError("facenet-pytorch not installed")
            self.net = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        elif backend == "insightface":
            if FaceAnalysis is None:
                raise RuntimeError("insightface not installed")
            self.net = FaceAnalysis(allowed_modules=['recognition'])
            self.net.prepare(ctx_id=0)
        else:
            raise ValueError("Unknown backend")

    def get_embedding(self, face_img_rgb):
        # face_img_rgb: numpy HWC RGB normalized to [0,1] or preprocessed depending
        if self.backend == "facenet":
            import torch
            # expecting CHW tensor
            if len(face_img_rgb.shape)==3 and face_img_rgb.shape[2]==3:
                # HWC -> CHW
                x = np.transpose(face_img_rgb, (2,0,1))
            else:
                x = face_img_rgb
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = self.net(x)
            emb = emb.cpu().numpy().astype(np.float32).reshape(-1)
            # L2 normalize
            emb = emb / np.linalg.norm(emb)
            return emb
        else:
            # insightface extraction expects BGR or RGB depending; show example
            # Here we expect a HWC RGB uint8 image
            faces = self.net.get(np.asarray(face_img_rgb))
            if len(faces)==0:
                return None
            emb = faces[0].embedding
            emb = emb / np.linalg.norm(emb)
            return emb
