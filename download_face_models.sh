#!/usr/bin/env bash
# download_face_models.sh — fetch the YuNet + SFace ONNX models from OpenCV Zoo
# into ./models/ for face_engine.py.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
mkdir -p "$DIR"

BASE="https://github.com/opencv/opencv_zoo/raw/main/models"
YUNET="$BASE/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE="$BASE/face_recognition_sface/face_recognition_sface_2021dec.onnx"

echo "Downloading YuNet (face detection)..."
curl -fL "$YUNET" -o "$DIR/face_detection_yunet_2023mar.onnx"

echo "Downloading SFace (face recognition)..."
curl -fL "$SFACE" -o "$DIR/face_recognition_sface_2021dec.onnx"

echo
echo "Done. Models in: $DIR"
ls -lh "$DIR"
