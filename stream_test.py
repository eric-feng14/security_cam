#!/usr/bin/env python3
"""
stream_test.py — IMX500 detection with live MJPEG stream.
No notifications. Use this to visually verify bounding boxes before
moving on to the full app.
Open http://<your-pi-ip>:8080 in a browser on the same network.
"""

import time
import threading
import cv2
from flask import Flask, Response
from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# ── Config ────────────────────────────────────────────────────────────────────
MODEL      = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
CONFIDENCE = 0.45   # lower than the main app to see more detections during testing
PORT       = 8080
PERSON_IDX = 0      # adjust if diagnostic.py showed a different index for 'person'

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
        main={"size": (1280, 720), "format": "RGB888"},  # BGR = OpenCV native, no conversion needed
        controls={"FrameRate": 30},                       # stream at 30fps; IMX500 infers asynchronously
        buffer_count=12,
    )
)

# ── Shared state ──────────────────────────────────────────────────────────────
app        = Flask(__name__)
frame_lock = threading.Lock()
latest_jpg : bytes | None = None

# ── Detection ─────────────────────────────────────────────────────────────────
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


def draw_boxes(frame, detections: list) -> None:
    for d in detections:
        x, y, w, h = (int(v) for v in d.box)
        colour = (0, 200, 60) if d.category == PERSON_IDX else (0, 160, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(
            frame, f"{d.label}  {d.score:.0%}",
            (x, max(y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2,
        )

# ── Per-frame callback ────────────────────────────────────────────────────────
def on_frame(request):
    global latest_jpg
    metadata   = request.get_metadata()
    detections = parse_detections(metadata)

    with MappedArray(request, "main") as m:
        draw_boxes(m.array, detections)
        ok, enc = cv2.imencode(".jpg", m.array, [cv2.IMWRITE_JPEG_QUALITY, 80])  # already BGR
        if ok:
            with frame_lock:
                latest_jpg = enc.tobytes()

    # Also print person detections to the terminal so you can see both at once
    for d in detections:
        if d.category == PERSON_IDX:
            print(f"[person]  score={d.score:.2f}  box={[int(v) for v in d.box]}")


picam2.pre_callback = on_frame

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
  <title>Stream Test</title>
  <style>
    body { background:#111; display:flex; flex-direction:column;
           align-items:center; padding:2rem; font-family:system-ui,sans-serif; }
    h2   { color:#4fc3f7; margin-bottom:1rem; }
    img  { max-width:min(100%, 900px); border-radius:8px; border:1px solid #333; }
  </style>
</head>
<body>
  <h2>📷 Stream Test — IMX500 Detection</h2>
  <img src="/stream" alt="live feed">
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Stream live at  http://<your-pi-ip>:{PORT}")
    print("Person detections will also print here. Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
