# Pi Security Cam — IMX500 Person Detection + Face Recognition

A Raspberry Pi 5 security camera built on the **Sony IMX500 AI Camera**. The
IMX500 runs person detection *on the sensor*, while the Pi's CPU runs face
recognition to label who's in frame. A live MJPEG dashboard shows the feed with
bounding boxes and name labels.

> **Status:** detection + live stream + face recognition (labelling only).
> Push notifications and known/unknown alert logic are not wired up yet.

---

## How it works

```
IMX500 (on-sensor)            Pi 5 CPU (background thread)
─────────────────             ───────────────────────────
person detection   ──frame──▶ YuNet (detect faces)
SSD MobileNetV2               └▶ SFace (128-d embedding)
                                 └▶ cosine match vs faces.db ─▶ name / "unknown"
```

The IMX500 holds only one model, so person detection lives on the sensor and
face recognition runs on the ARM cores. Recognition runs in its own thread, so
it never throttles the 30fps video stream — the camera callback just draws the
most recent results.

---

## Files

| File | Purpose |
|---|---|
| `diagnostic.py` | Print raw IMX500 output tensors — verify box/score/class order |
| `stream_test.py` | Live stream with **person** boxes only (no face recognition) |
| `download_face_models.sh` | Download the YuNet + SFace ONNX models into `models/` |
| `face_db.py` | SQLite store (`faces.db`) for enrolled face embeddings |
| `face_engine.py` | Face detection + embedding + matching (the recognition core) |
| `enroll_faces.py` | Register a person's face into the database |
| `manage_faces.py` | List / remove enrolled people |
| `stream_faces.py` | Live stream with **person** boxes **+ face name labels** |
| `faces.db` | *(auto-created)* the enrolled-faces database |
| `models/` | *(auto-created)* downloaded ONNX models |

---

## Prerequisites

A Raspberry Pi 5 with the IMX500 AI Camera, set up through **Step 5b** of the
build procedure. In short, the system packages and virtual environment:

```bash
sudo apt install -y python3-picamera2 imx500-all python3-opencv python3-pip python3-venv
python3 -m venv ~/security_cam/venv --system-site-packages
source ~/security_cam/venv/bin/activate
pip install flask requests
```

`picamera2`, `opencv`, and `numpy` come from `apt` (system packages); `flask`
lives in the venv. **Activate the venv in every new SSH session:**

```bash
source ~/security_cam/venv/bin/activate
```

---

## First-time setup

Run once, from `~/security_cam` with the venv active:

```bash
chmod +x download_face_models.sh
./download_face_models.sh
```

This fetches two small ONNX models from OpenCV Zoo into `models/`:

- `face_detection_yunet_2023mar.onnx` — locates faces + 5 landmarks
- `face_recognition_sface_2021dec.onnx` — produces the 128-d face embedding

---

## Usage

### 1. Verify the camera + detection (optional sanity checks)

```bash
python diagnostic.py        # confirm tensor order; look for [person] score=0.XX
python stream_test.py       # person boxes only — open http://<pi-ip>:8080
```

### 2. Enroll known faces

Register each person you want recognised. Stand ~1m away, well lit, looking at
the lens; move your head slightly between captures for a more robust match.

```bash
python enroll_faces.py "Eric"                 # 5 samples by default
python enroll_faces.py "Sam" --samples 8      # more samples = more robust
```

Options: `--samples N` (how many good captures), `--interval S` (seconds between
attempts). Progress prints to the terminal, so this works over SSH — no display
needed. Re-running for an existing name **adds** more samples.

### 3. Manage the database

```bash
python manage_faces.py list                   # who's enrolled + sample counts
python manage_faces.py remove "Eric"          # delete a person (asks to confirm)
python manage_faces.py remove "Eric" -y       # delete without the prompt
```

Names are **case-sensitive** and must match how you enrolled — use `list` to see
the exact strings.

### 4. Run the recognition stream

```bash
python stream_faces.py
```

Open **`http://<pi-ip>:8080`** in a browser on the same network. Find the Pi's
IP with `hostname -I`.

You'll see:
- **Green box + name + similarity %** — an enrolled person
- **Red box + "unknown"** — a face that matched nobody
- **Dim grey box** — the IMX500 person detection (kept for continuity)
- Terminal also prints `[face]  Eric  sim=0.71  box=...`

> The engine loads the database **at startup**. After enrolling or removing
> anyone, restart `stream_faces.py` to pick up the change.

---

## Tuning

Edit the constants at the top of `face_engine.py`:

| Constant | Default | Effect |
|---|---|---|
| `COSINE_THRESHOLD` | `0.363` | Match strictness. Known people tagged "unknown" → **lower** (~0.30). Strangers matching someone → **raise** (~0.40). |
| `DETECT_SCORE_THRESHOLD` | `0.85` | How confident a *face detection* must be before it's considered at all. |

`0.363` is SFace's standard same-identity cosine threshold — a good starting
point. After changing either value, restart `stream_faces.py`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: ... .onnx` | Run `./download_face_models.sh` |
| `ModuleNotFoundError: picamera2` / `cv2` | Activate the venv; these come from `apt` |
| No face boxes at all | Improve lighting; lower `DETECT_SCORE_THRESHOLD`; check models downloaded |
| Known person shows "unknown" | Enroll more samples; lower `COSINE_THRESHOLD` |
| Stranger matches a known name | Raise `COSINE_THRESHOLD` |
| Changes to the DB not reflected | Restart `stream_faces.py` (DB is loaded at startup) |
| Dashboard loads but no image | Confirm port 8080 reachable on LAN; `curl http://localhost:8080/stream` |

---

## What's next

- Known/unknown **alert logic** (e.g. notify only on strangers)
- **ntfy** Android push notifications with snapshots
- Run on boot via **systemd**
