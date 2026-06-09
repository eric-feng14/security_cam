#!/usr/bin/env python3
"""
face_engine.py — Face detection + recognition on the Pi CPU.

Uses two small ONNX models from OpenCV Zoo (run `download_face_models.sh`):
  - YuNet  (cv2.FaceDetectorYN)   → locates faces + 5 landmarks
  - SFace  (cv2.FaceRecognizerSF) → 128-d embedding per aligned face

Matching is brute-force cosine similarity against the embeddings enrolled
in faces.db. With only a handful of people this is effectively free.

The IMX500 runs person detection on-sensor; this module runs on the ARM
cores, so both pipelines work at once without fighting over the sensor.
"""

import os
import numpy as np
import cv2

import face_db

MODELS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DETECTOR_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# SFace's recommended same-identity cosine threshold. Higher = stricter.
COSINE_THRESHOLD = 0.363
# YuNet detection confidence (0-1). Raise to drop weak/false faces.
DETECT_SCORE_THRESHOLD = 0.85


class FaceEngine:
    """Detect faces, embed them, and match against the enrolled database."""

    def __init__(
        self,
        detector_model: str = DETECTOR_MODEL,
        recognizer_model: str = RECOGNIZER_MODEL,
        detect_score_threshold: float = DETECT_SCORE_THRESHOLD,
        cosine_threshold: float = COSINE_THRESHOLD,
    ):
        for path in (detector_model, recognizer_model):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model not found: {path}\n"
                    "Run ./download_face_models.sh to fetch the ONNX models."
                )

        # input_size is updated per-frame in detect(); (320, 320) is a placeholder.
        self.detector = cv2.FaceDetectorYN.create(
            detector_model, "", (320, 320),
            score_threshold=detect_score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_model, "")
        self.cosine_threshold = cosine_threshold

        # Enrolled faces, pre-normalised for fast cosine compares.
        self.known: list[tuple[str, np.ndarray]] = []
        self.load_known()

    # ── enrolled database ───────────────────────────────────────────────────
    def load_known(self) -> None:
        """(Re)load enrolled embeddings from faces.db, L2-normalised."""
        self.known = []
        for name, emb in face_db.load_all():
            norm = np.linalg.norm(emb)
            if norm > 0:
                self.known.append((name, emb / norm))

    @property
    def known_names(self) -> list[str]:
        return sorted({name for name, _ in self.known})

    # ── detection / embedding ───────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        """Return YuNet detections (N x 15) for `frame`, or None if no faces."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces  # None, or rows of [x, y, w, h, 5 landmarks (10), score]

    def embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """Align one detected face and return its 128-d embedding."""
        aligned = self.recognizer.alignCrop(frame, face_row)
        return self.recognizer.feature(aligned).flatten()

    # ── matching ────────────────────────────────────────────────────────────
    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        """
        Compare an embedding to all enrolled faces.
        Returns (name, similarity) — name is "unknown" if nothing clears
        the cosine threshold (or if nobody is enrolled yet).
        """
        norm = np.linalg.norm(embedding)
        if norm == 0 or not self.known:
            return "unknown", 0.0
        query = embedding / norm

        best_name, best_sim = "unknown", -1.0
        for name, known_vec in self.known:
            sim = float(np.dot(query, known_vec))
            if sim > best_sim:
                best_sim, best_name = sim, name

        if best_sim >= self.cosine_threshold:
            return best_name, best_sim
        return "unknown", max(best_sim, 0.0)

    def identify(self, frame: np.ndarray) -> list[dict]:
        """
        Full pipeline on one frame.
        Returns a list of {box: (x, y, w, h), name: str, score: float}.
        """
        faces = self.detect(frame)
        if faces is None:
            return []

        results = []
        for face_row in faces:
            embedding = self.embed(frame, face_row)
            name, sim = self.match(embedding)
            x, y, w, h = (int(v) for v in face_row[:4])
            results.append({"box": (x, y, w, h), "name": name, "score": sim})
        return results
