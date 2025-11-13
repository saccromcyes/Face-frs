from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import numpy as np
import io

Base = declarative_base()

class Identity(Base):
    __tablename__ = "identities"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    image_path = Column(String, nullable=True)
    embedding = Column(LargeBinary, nullable=False)  # store numpy array bytes
    added_on = Column(DateTime, default=datetime.datetime.utcnow)
    meta_info = Column(String, nullable=True)  

# DB helper
def get_engine(db_path="face_gallery.db"):
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    return engine

def create_db(db_path="face_gallery.db"):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine

# utils for embedding bytes
def embed_to_bytes(arr: np.ndarray) -> bytes:
    memfile = io.BytesIO()
    np.save(memfile, arr)
    memfile.seek(0)
    return memfile.read()

def bytes_to_embed(b: bytes) -> np.ndarray:
    memfile = io.BytesIO(b)
    memfile.seek(0)
    return np.load(memfile)

