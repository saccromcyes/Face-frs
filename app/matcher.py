
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Matcher:
    def __init__(self, db_session, top_k=5, threshold=0.45):
        self.db_session = db_session
        self.top_k = top_k
        self.threshold = threshold

    def load_all_embeddings(self):
        from models import Identity, bytes_to_embed
        ids = []
        names = []
        embs = []
        for row in self.db_session.query(Identity).all():
            ids.append(row.id)
            names.append(row.name)
            embs.append(bytes_to_embed(row.embedding))
        if len(embs)>0:
            embs = np.vstack(embs)
        else:
            embs = np.zeros((0,512), dtype=np.float32)
        return ids, names, embs

    def match(self, query_emb):
        ids, names, embs = self.load_all_embeddings()
        if embs.shape[0]==0:
            return []
        sims = cosine_similarity(query_emb.reshape(1,-1), embs).reshape(-1)  # cosine similarity in [-1,1]
        idx_sorted = np.argsort(-sims)[:self.top_k]
        results = []
        for idx in idx_sorted:
            score = float(sims[idx])
            if score < (1 - self.threshold):  # if threshold is distance; adjust semantics as needed
                # if using cosine similarity, threshold interpretation differs; here assume high=good
                pass
            results.append({"id": ids[idx], "name": names[idx], "score": score})
        return results
