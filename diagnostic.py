#!/usr/bin/env python3
"""
diagnostic.py — Print raw IMX500 output tensors so you can verify
the index order (boxes / scores / classes) before running the main app.
Stand in front of the camera and watch the terminal output.
"""

import time
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
THRESHOLD = 0.35  # low threshold so we see plenty of output

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

imx500     = IMX500(MODEL)
intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
picam2     = Picamera2(imx500.camera_num)

picam2.start(
    picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate or 30},
        buffer_count=12,
    )
)

print("Running — point camera at yourself and press Ctrl+C to stop\n")
shapes_printed = False

try:
    while True:
        metadata = picam2.capture_metadata()
        outputs  = imx500.get_outputs(metadata, add_batch=True)
        if outputs is None:
            time.sleep(0.1)
            continue

        # Print tensor shapes on first valid frame
        if not shapes_printed:
            print("=== Output tensor shapes ===")
            for i, o in enumerate(outputs):
                print(f"  outputs[{i}]: shape={o.shape}  dtype={o.dtype}")
            print()
            shapes_printed = True

        # Assuming order: boxes=0, scores=1, classes=2
        boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]

        for box, score, cls in zip(boxes, scores, classes):
            if float(score) >= THRESHOLD:
                idx   = int(cls)
                label = COCO_LABELS[idx] if idx < len(COCO_LABELS) else f"class_{idx}"
                print(f"  [{label}]  class_idx={idx}  score={float(score):.2f}  "
                      f"box=[{', '.join(f'{float(v):.3f}' for v in box)}]")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    picam2.stop()
