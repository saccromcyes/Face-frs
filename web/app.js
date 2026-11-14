// FaceFenix in the browser: detection, landmarks, 128-d face descriptors and a blink liveness check,
// all with TensorFlow.js on the visitor's device. Nothing is uploaded; face prints stay in localStorage.
import * as faceapi from "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/dist/face-api.esm.js";

const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/";
const STORE_KEY = "facefenix-people";
const SAMPLES = 5;
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const video = $("#video");
const photo = $("#photo");
const overlay = $("#overlay");
const ctx = overlay.getContext("2d");
const detector = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

const state = {
  people: loadPeople(),
  threshold: 0.5,
  mode: null, // "camera" | "photo"
  running: false,
  lastFaces: [],
  enrol: null, // { name, samples: [], lastAt }
  live: null, // { start, blinks, closed, series: [{t, ear}] }
  frames: [],
};

// ---------------------------------------------------------------- storage
function loadPeople() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE_KEY) || "[]");
    return raw.filter((p) => p && typeof p.name === "string" && Array.isArray(p.descriptor) && p.descriptor.length === 128)
      .map((p) => ({ ...p, descriptor: Float32Array.from(p.descriptor) }));
  } catch (_) { return []; }
}
function savePeople() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(state.people.map((p) => ({ ...p, descriptor: Array.from(p.descriptor) }))));
  } catch (_) { /* private mode: keep in memory for this visit */ }
}

// ---------------------------------------------------------------- models
async function init() {
  try {
    try { await faceapi.tf.setBackend("webgl"); } catch (_) { await faceapi.tf.setBackend("cpu"); }
    await faceapi.tf.ready();
    $("#m-backend").textContent = `Backend ${faceapi.tf.getBackend() === "webgl" ? "WebGL (GPU)" : "CPU"}`;
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
    ]);
    $("#load-status").textContent = "Models ready. Start the camera, or try a photo.";
    $("#start-camera").disabled = false;
    $("#upload").disabled = false;
    $("#upload-label").removeAttribute("aria-disabled");
    renderPeople();
  } catch (err) {
    $("#load-status").textContent = "The face models couldn't load. Check your connection and reload the page.";
    console.error(err);
  }
}

// ---------------------------------------------------------------- sources
async function startCamera() {
  hideError();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    showError(err && err.name === "NotAllowedError"
      ? "Camera access was blocked. Allow it in your browser's site settings, or use a photo instead."
      : "No camera was found. You can use a photo instead.");
    return;
  }
  state.mode = "camera";
  photo.hidden = true;
  video.hidden = false;
  $("#stage-empty").hidden = true;
  $("#stop").hidden = false;
  $("#upload2-label").hidden = false;
  setControls();
  state.running = true;
  loop();
}

function stopCamera() {
  state.running = false;
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  state.mode = null;
  state.lastFaces = [];
  stopLiveness(null);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  $("#stage-empty").hidden = false;
  $("#stop").hidden = true;
  $("#upload2-label").hidden = true;
  setControls();
  renderPeople();
}

async function usePhoto(file) {
  if (!file) return;
  hideError();
  if (state.mode === "camera") stopCamera();
  const url = URL.createObjectURL(file);
  photo.src = url;
  await photo.decode().catch(() => null);
  if (!photo.naturalWidth) { showError("That file couldn't be read as an image."); return; }
  state.mode = "photo";
  video.hidden = true;
  photo.hidden = false;
  $("#stage-empty").hidden = true;
  $("#upload2-label").hidden = false;
  setControls();
  const t0 = performance.now();
  const faces = await faceapi.detectAllFaces(photo, detector).withFaceLandmarks().withFaceDescriptors();
  handleFrame(faces, performance.now() - t0, photo.naturalWidth, photo.naturalHeight, false);
  if (!faces.length) showError("No face found in that photo. Try one where the face is large and well lit.");
}

// ---------------------------------------------------------------- the loop
async function loop() {
  if (!state.running) return;
  if (video.readyState >= 2 && video.videoWidth) {
    const t0 = performance.now();
    const faces = await faceapi.detectAllFaces(video, detector).withFaceLandmarks().withFaceDescriptors();
    if (!state.running) return;
    handleFrame(faces, performance.now() - t0, video.videoWidth, video.videoHeight, true);
  }
  requestAnimationFrame(loop);
}

function bestMatch(descriptor) {
  let best = null;
  for (const p of state.people) {
    const d = faceapi.euclideanDistance(descriptor, p.descriptor);
    if (!best || d < best.distance) best = { person: p, distance: d };
  }
  return best;
}

function handleFrame(faces, ms, w, h, mirrored) {
  // metrics
  const now = performance.now();
  state.frames = state.frames.filter((t) => now - t < 1000).concat(now);
  $("#m-ms").textContent = `${Math.round(ms)} ms per frame`;
  $("#m-fps").textContent = state.mode === "camera" ? `${state.frames.length} fps` : "single image";
  $("#m-faces").textContent = `${faces.length} face${faces.length === 1 ? "" : "s"}`;

  // largest face first: it drives enrolment, the distance list and liveness
  faces.sort((a, b) => b.detection.box.area - a.detection.box.area);
  state.lastFaces = faces;
  draw(faces, w, h, mirrored);
  if (state.enrol) collectSample(faces);
  if (state.live) trackBlink(faces[0]);
  renderPeople();
}

function draw(faces, w, h, mirrored) {
  if (overlay.width !== w || overlay.height !== h) { overlay.width = w; overlay.height = h; }
  ctx.clearRect(0, 0, w, h);
  const lw = Math.max(2, Math.round(w / 240));
  ctx.lineWidth = lw;
  ctx.font = `700 ${Math.max(14, Math.round(w / 34))}px Manrope, system-ui, sans-serif`;
  for (const f of faces) {
    const b = f.detection.box;
    const x = mirrored ? w - b.x - b.width : b.x;
    const m = state.people.length ? bestMatch(f.descriptor) : null;
    const known = m && m.distance <= state.threshold;
    const color = known ? css("--match") : state.people.length ? css("--unknown") : css("--lens");
    ctx.strokeStyle = color;
    corners(x, b.y, b.width, b.height, lw);
    const label = known ? `${m.person.name}  ${m.distance.toFixed(2)}` : state.people.length ? `Unknown  ${m.distance.toFixed(2)}` : `Face ${Math.round(f.detection.score * 100)}%`;
    const tw = ctx.measureText(label).width + 12;
    const th = parseInt(ctx.font.match(/\d+(?=px)/)[0], 10) + 10;
    const ly = b.y > th + 4 ? b.y - th - 4 : b.y + b.height + 4;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 6, ly + th - 8);
    // eyes, so the liveness signal is visible
    ctx.fillStyle = color;
    for (const pt of [...f.landmarks.getLeftEye(), ...f.landmarks.getRightEye()]) {
      const px = mirrored ? w - pt.x : pt.x;
      ctx.fillRect(px - lw / 2, pt.y - lw / 2, lw, lw);
    }
  }
}

function corners(x, y, w, h, lw) {
  const c = Math.min(w, h) * 0.22;
  ctx.beginPath();
  ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y);
  ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c);
  ctx.moveTo(x + w, y + h - c); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - c, y + h);
  ctx.moveTo(x + c, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - c);
  ctx.stroke();
  ctx.lineWidth = lw;
}

// ---------------------------------------------------------------- enrolment
function startEnrol(name) {
  hideError();
  if (state.people.some((p) => p.name.toLowerCase() === name.toLowerCase())) {
    $("#enrol-note").textContent = `${name} is already enrolled. Forget them first to re-enrol.`;
    return;
  }
  if (state.mode === "photo") {
    const faces = state.lastFaces;
    if (faces.length !== 1) { $("#enrol-note").textContent = faces.length ? "The photo has several faces. Use a photo with exactly one." : "No face in the photo to enrol."; return; }
    addPerson(name, [faces[0].descriptor], "a photo");
    return;
  }
  state.enrol = { name, samples: [], lastAt: 0 };
  $("#enrol-progress").hidden = false;
  $("#enrol-progress b").style.width = "0%";
  $("#enrol-note").textContent = "Look at the camera and turn your head slightly between samples.";
  setControls();
}

function collectSample(faces) {
  const e = state.enrol;
  const now = performance.now();
  if (faces.length !== 1) { $("#enrol-note").textContent = faces.length ? "Only one face in view, please." : "No face in view."; return; }
  if (faces[0].detection.score < 0.7 || now - e.lastAt < 350) return;
  e.samples.push(faces[0].descriptor);
  e.lastAt = now;
  $("#enrol-progress b").style.width = `${(e.samples.length / SAMPLES) * 100}%`;
  $("#enrol-note").textContent = `Sample ${e.samples.length} of ${SAMPLES}`;
  if (e.samples.length >= SAMPLES) {
    state.enrol = null;
    addPerson(e.name, e.samples, `${SAMPLES} camera samples`);
    setTimeout(() => { $("#enrol-progress").hidden = true; }, 600);
  }
}

function addPerson(name, descriptors, source) {
  const avg = new Float32Array(128);
  for (const d of descriptors) for (let i = 0; i < 128; i++) avg[i] += d[i] / descriptors.length;
  state.people.push({ name, descriptor: avg, source, created: new Date().toISOString() });
  savePeople();
  $("#enrol-name").value = "";
  $("#enrol-note").textContent = `Enrolled ${name} from ${source}. Only the 128-number face print was kept.`;
  setControls();
  renderPeople();
}

// ---------------------------------------------------------------- recognise list
function renderPeople() {
  const list = $("#people");
  if (!state.people.length) { list.innerHTML = `<li class="empty">Nobody enrolled yet.</li>`; return; }
  const face = state.lastFaces[0];
  list.innerHTML = state.people.map((p, i) => {
    const d = face ? faceapi.euclideanDistance(face.descriptor, p.descriptor) : null;
    const match = d !== null && d <= state.threshold;
    const width = d === null ? 0 : Math.max(0, Math.min(100, (1 - d) * 100));
    return `<li class="${match ? "match" : ""}"><span class="name">${esc(p.name)}</span>
      <span class="dist">${d === null ? "no face in view" : `${match ? "match, " : ""}distance ${d.toFixed(2)}`}</span>
      <button type="button" class="forget" data-i="${i}">Forget</button>
      <span class="bar" aria-hidden="true"><b style="width:${width}%"></b></span></li>`;
  }).join("");
}
$("#people").addEventListener("click", (e) => {
  const b = e.target.closest(".forget");
  if (!b) return;
  state.people.splice(Number(b.dataset.i), 1);
  savePeople();
  renderPeople();
});

// ---------------------------------------------------------------- liveness
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const eyeRatio = (p) => (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]));

function startLiveness() {
  state.live = { start: performance.now(), blinks: 0, closed: false, series: [] };
  $("#live-result").className = "live-result";
  $("#live-result").textContent = "Blink twice…";
  setControls();
}

function trackBlink(face) {
  const L = state.live;
  const now = performance.now();
  if (face) {
    const ear = (eyeRatio(face.landmarks.getLeftEye()) + eyeRatio(face.landmarks.getRightEye())) / 2;
    L.series.push({ t: now, ear });
    // Thresholds relative to this person's open-eye level, so they adapt to face shape and camera angle.
    const recent = L.series.slice(-45).map((s) => s.ear).sort((a, b) => a - b);
    const open = recent[Math.floor(recent.length * 0.8)];
    if (L.series.length > 5) {
      if (!L.closed && ear < open * 0.72) L.closed = true;
      else if (L.closed && ear > open * 0.88) { L.closed = false; L.blinks += 1; }
    }
  }
  drawEar(L, now);
  if (L.blinks >= 2) stopLiveness(true);
  else if (now - L.start > 10000) stopLiveness(false);
  else $("#live-result").textContent = `Blink twice… ${L.blinks} so far, ${Math.ceil((10000 - (now - L.start)) / 1000)} s left`;
}

function drawEar(L, now) {
  const c = $("#ear");
  const g = c.getContext("2d");
  const w = c.width, h = c.height;
  g.clearRect(0, 0, w, h);
  const pts = L.series.filter((s) => now - s.t < 10000);
  if (pts.length < 2) return;
  const max = Math.max(...pts.map((p) => p.ear)) * 1.1;
  g.strokeStyle = css("--lens");
  g.lineWidth = 2;
  g.beginPath();
  pts.forEach((p, i) => {
    const x = ((p.t - (now - 10000)) / 10000) * w;
    const y = h - (p.ear / max) * (h - 6) - 3;
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
}

function stopLiveness(passed) {
  if (!state.live) return;
  state.live = null;
  const r = $("#live-result");
  if (passed === true) { r.className = "live-result pass"; r.textContent = "Live person detected: two blinks."; }
  else if (passed === false) { r.className = "live-result fail"; r.textContent = "No blinks seen in ten seconds. A photo would fail here too. Try again."; }
  else { r.className = "live-result"; r.textContent = ""; }
  setControls();
}

// ---------------------------------------------------------------- wiring
function setControls() {
  const ready = !$("#start-camera").disabled || state.mode;
  $("#enrol-btn").disabled = !state.mode || !!state.enrol;
  $("#live-btn").disabled = state.mode !== "camera" || !!state.live;
  $("#live-btn").title = state.mode === "photo" ? "Liveness needs the camera." : "";
  return ready;
}
function showError(msg) { $("#error").textContent = msg; $("#error").hidden = false; }
function hideError() { $("#error").hidden = true; }

$("#start-camera").addEventListener("click", startCamera);
$("#stop").addEventListener("click", stopCamera);
$("#upload").addEventListener("change", (e) => { usePhoto(e.target.files[0]); e.target.value = ""; });
$("#upload2").addEventListener("change", (e) => { usePhoto(e.target.files[0]); e.target.value = ""; });
$("#enrol-form").addEventListener("submit", (e) => { e.preventDefault(); const n = $("#enrol-name").value.trim(); if (n) startEnrol(n); });
$("#live-btn").addEventListener("click", startLiveness);
$("#threshold").addEventListener("input", (e) => {
  state.threshold = Number(e.target.value);
  $("#th-out").textContent = state.threshold.toFixed(2);
  if (state.mode === "photo" && state.lastFaces.length) draw(state.lastFaces, photo.naturalWidth, photo.naturalHeight, false);
  renderPeople();
});
document.addEventListener("visibilitychange", () => { if (document.hidden && state.mode === "camera") stopCamera(); });

// Test hook: ?debug exposes state to automated browser tests.
if (new URLSearchParams(location.search).has("debug")) window.__facefenix = state;

init();
