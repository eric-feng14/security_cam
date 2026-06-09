#!/usr/bin/env python3
"""
enroll_faces.py — Register a known person's face into faces.db.

Captures several frames from the camera, keeps only the ones with exactly
one clearly-detected face, embeds them with SFace, and stores them under the
given name. Run once per person.

    python enroll_faces.py "Eric"
    python enroll_faces.py "Eric" --samples 8 --interval 0.8

Stand ~1m from the camera, well lit, looking roughly at the lens. Move your
head slightly between captures (small angle changes) for a more robust match.
No display is needed — progress prints to the terminal, so this works over SSH.
"""

import argparse
import time

from picamera2 import Picamera2

import face_db
from face_engine import FaceEngine


def main() -> None:
    ap = argparse.ArgumentParser(description="Enroll a face into faces.db")
    ap.add_argument("name", help="Person's name, e.g. \"Eric\"")
    ap.add_argument("--samples", type=int, default=5,
                    help="How many good samples to capture (default 5)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="Seconds between capture attempts (default 1.0)")
    args = ap.parse_args()

    face_db.init_db()
    engine = FaceEngine()

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"},  # BGR for OpenCV
        )
    )
    picam2.start()
    time.sleep(1.0)  # let auto-exposure settle

    existing = face_db.counts().get(args.name, 0)
    if existing:
        print(f"Note: '{args.name}' already has {existing} sample(s); "
              f"new captures will be added.")

    print(f"\nEnrolling '{args.name}' — capturing {args.samples} sample(s). "
          f"Look at the camera.\n")

    collected = 0
    try:
        while collected < args.samples:
            frame = picam2.capture_array("main")
            faces = engine.detect(frame)
            n = 0 if faces is None else len(faces)

            if n == 0:
                print("  no face detected — move into frame / improve lighting")
            elif n > 1:
                print(f"  {n} faces detected — only one person should be in frame")
            else:
                embedding = engine.embed(frame, faces[0])
                face_db.add_embedding(args.name, embedding)
                collected += 1
                score = float(faces[0][-1])
                print(f"  captured {collected}/{args.samples}  "
                      f"(face score {score:.2f})")

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        picam2.stop()

    total = face_db.counts().get(args.name, 0)
    print(f"\nDone. '{args.name}' now has {total} sample(s) in faces.db.")


if __name__ == "__main__":
    main()
