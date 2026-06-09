#!/usr/bin/env python3
"""
stream_faces.py — IMX500 person detection + OpenCV face recognition.

Step 5b's live stream, now with face labels: each detected face is boxed and
labelled with the enrolled person's name, or "unknown" if no match.
No notifications yet — this step is purely to verify recognition visually.

Enroll people first with:  python enroll_faces.py "Name"
Then open               :  http://<your-pi-ip>:8080

Face recognition runs in a background thread (not the camera callback) so it
never throttles the stream — the callback just draws the most recent results.
"""

import time
import threading
import cv2
from flask import Flask, Response
from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

import face_db
from face_engine import FaceEngine

# ── Config ────────────────────────────────────────────────────────────────────
MODEL      = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
CONFIDENCE = 0.45   # IMX500 person-detection threshold
PORT       = 8080
PERSON_IDX = 0      # COCO class index for 'person'

COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush",
]

# ── Camera init ───────────────────────────────────────────────────────────────
imx500     = IMX500(MODEL)
intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
picam2     = Picamera2(imx500.camera_num)

imx500.show_network_fw_progress_bar()
picam2.start(
    picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"},  # BGR = OpenCV native
        controls={"FrameRate": 30},
        buffer_count=12,
    )
)

# ── Face engine ───────────────────────────────────────────────────────────────
face_db.init_db()
engine = FaceEngine()
print(f"Loaded {len(engine.known)} embedding(s) for: "
      f"{', '.join(engine.known_names) or '(nobody enrolled yet)'}")

# ── Shared state ──────────────────────────────────────────────────────────────
app          = Flask(__name__)
frame_lock   = threading.Lock()
latest_jpg   : bytes | None = None
raw_lock     = threading.Lock()
latest_raw   = None                # most recent BGR frame for the recog thread
results_lock = threading.Lock()
face_results : list = []           # most recent [{box, name, score}, ...]

# ── IMX500 person detection ───────────────────────────────────────────────────
class Detection:
    def __init__(self, box, category: int, score: float, metadata: dict):
        self.category = category
        self.score    = score
        self.box      = imx500.convert_inference_coords(box, metadata, picam2)

    @property
    def label(self) -> str:
        return COCO_LABELS[self.category] if self.category < len(COCO_LABELS) \
               else f"class_{self.category}"


def parse_detections(metadata: dict) -> list:
    outputs = imx500.get_outputs(metadata, add_batch=True)
    if outputs is None:
        return []
    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
    return [
        Detection(box, int(cls), float(score), metadata)
        for box, score, cls in zip(boxes, scores, classes)
        if float(score) >= CONFIDENCE
    ]


def draw_person_boxes(frame, detections: list) -> None:
    # Person boxes are drawn dim/grey so the coloured face labels stand out.
    for d in detections:
        if d.category != PERSON_IDX:
            continue
        x, y, w, h = (int(v) for v in d.box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 120), 1)


def draw_face_labels(frame, results: list) -> None:
    for r in results:
        x, y, w, h = r["box"]
        known  = r["name"] != "unknown"
        colour = (0, 200, 60) if known else (0, 0, 255)   # green / red (BGR)
        label  = f"{r['name']}  {r['score']:.0%}" if known else "unknown"
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(
            frame, label, (x, max(y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
        )

# ── Per-frame callback (camera thread) — keep this fast ───────────────────────
def on_frame(request):
    global latest_jpg, latest_raw
    metadata   = request.get_metadata()
    detections = parse_detections(metadata)

    with MappedArray(request, "main") as m:
        # Hand a copy to the recognition thread.
        with raw_lock:
            latest_raw = m.array.copy()

        draw_person_boxes(m.array, detections)
        with results_lock:
            draw_face_labels(m.array, face_results)

        ok, enc = cv2.imencode(".jpg", m.array, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with frame_lock:
                latest_jpg = enc.tobytes()


picam2.pre_callback = on_frame

# ── Recognition worker (background thread) ────────────────────────────────────
def recognition_loop():
    global face_results
    while True:
        with raw_lock:
            frame = None if latest_raw is None else latest_raw.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        results = engine.identify(frame)
        with results_lock:
            face_results = results

        # Log recognised faces to the terminal as well.
        for r in results:
            print(f"[face] {r['name']:>12s}  sim={r['score']:.2f}  "
                  f"box={r['box']}")

        time.sleep(0.05)  # ~throttle; Pi 5 will run this ~10-15 fps


threading.Thread(target=recognition_loop, daemon=True).start()

# ── Flask MJPEG stream ────────────────────────────────────────────────────────
def mjpeg():
    while True:
        with frame_lock:
            frame = latest_jpg
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            time.sleep(0.04)

@app.route("/stream")
def stream():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Face Recognition Test</title>
  <style>
    body { background:#111; display:flex; flex-direction:column;
           align-items:center; padding:2rem; font-family:system-ui,sans-serif; }
    h2   { color:#4fc3f7; margin-bottom:1rem; }
    img  { max-width:min(100%, 900px); border-radius:8px; border:1px solid #333; }
  </style>
</head>
<body>
  <h2>🙂 Face Recognition Test — IMX500 + SFace</h2>
  <img src="/stream" alt="live feed">
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Stream live at  http://<your-pi-ip>:{PORT}")
    print("Recognised faces will also print here. Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
