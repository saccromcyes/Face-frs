# FaceFenix — face recognition that never leaves your browser

[![CI](https://github.com/saccromcyes/Face-frs/actions/workflows/ci.yml/badge.svg)](https://github.com/saccromcyes/Face-frs/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/live%20demo-GitHub%20Pages-0093B8)](https://saccromcyes.github.io/Face-frs/)

**[Try it →](https://saccromcyes.github.io/Face-frs/)** Enrol your face with the webcam (or a photo), watch it get recognised live, and pass a blink-based liveness check. Detection, landmarks, embeddings and matching all run on your device with TensorFlow.js: **no image or face data is ever uploaded.**

<!-- Screenshot added after the first deploy: docs/screenshot.png -->

## Two versions

| | In-browser demo (`web/`) | Python server (`app/`) |
|---|---|---|
| Detection | Tiny Face Detector | MTCNN |
| Landmarks | 68-point model | 5-point (MTCNN) |
| Embedding | 128-d ResNet-34-style network | 512-d InceptionResnetV1 (FaceNet, VGGFace2) |
| Matching | Euclidean distance, adjustable threshold | Cosine distance, `MATCH_THRESHOLD` |
| Liveness | Blink detection from the eye aspect ratio | — |
| Runs on | The visitor's GPU (WebGL) or CPU | FastAPI + Streamlit, ONNX export and CPU benchmarks in `tools/` |
| Privacy | Face prints (128 numbers) stay in the browser's storage; images are never stored | Embeddings in SQLite |

## How the browser version works

1. **Detect** faces in each frame with a 190 KB Tiny Face Detector.
2. **Locate** 68 landmarks per face.
3. **Embed** the aligned face as a 128-number descriptor.
4. **Match** by Euclidean distance against enrolled face prints. Enrolling from the camera averages five samples taken a moment apart, which is more stable than a single frame. The threshold slider shows the trade-off between false matches and missed matches.
5. **Liveness.** The eye aspect ratio (eye height ÷ width from the landmarks) dips during a blink. The thresholds adapt to each person's open-eye level, and two blinks within ten seconds pass. A printed photo or a still image on a phone fails.

The page also shows the backend in use (WebGL or CPU), milliseconds per frame and frames per second.

## Privacy

This repository previously included photos of real people and a database of their face embeddings. Both have been removed, and `.gitignore` now keeps `gallery_images/` and `*.db` files out of the repository. The browser demo has no server at all: the only network requests are the one-time model downloads from jsDelivr.

## Fixes in this version

- **Unknown faces were always matched to someone.** The old `Matcher` computed similarities but never applied its threshold (the check was a `pass`), and the API hard-coded a Euclidean cut-off of 0.8 while reporting `1 − distance` as a similarity. Matching now lives in one function, `app/matcher.py`. It uses cosine distance with the configurable `MATCH_THRESHOLD`, returns `None` for strangers, and is covered by tests.

## Run the Python server

```bash
pip install -r docker/requirements.txt
uvicorn app.main:app --port 8000      # API
streamlit run streamlit_app.py         # UI
python tools/convert_onnx.py           # export the embedder to ONNX
python tools/benchmark_cpu.py          # CPU latency benchmark
pytest tests                           # matcher tests (numpy only)
```

## Run the browser version locally

```bash
cd web && python -m http.server 8080   # then open http://localhost:8080
```

Camera access needs `localhost` or HTTPS. The demo is published to GitHub Pages by `.github/workflows/pages.yml` on every change to `web/`.

## Credits

Browser models and runtime: [@vladmandic/face-api](https://github.com/vladmandic/face-api) (MIT), built on TensorFlow.js. Server models: [facenet-pytorch](https://github.com/timesler/facenet-pytorch).
