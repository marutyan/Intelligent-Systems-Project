# Intelligent-Systems-Project · **no06**

[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20|%203.10-blue)
![Django](https://img.shields.io/badge/Django-5.2-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-pose&nbsp;%7C&nbsp;det-red)

> **no06** converts raw videos into pose-based colour features, clusters them with t-SNE + k-means and offers a smooth, single-page UX for exploration.

---

## ✨ Features

| Area | What you get |
|------|--------------|
| **Upload & storage** | AJAX form, metadata in SQLite (`Video`), files under `no06/media/`. |
| **Feature extraction** | 1 fps sampling → YOLOv8-Pose on shoulders & hips → 12-dim RGB vector. |
| **Clustering** | t-SNE (3-D) + k-means (k = 5) rendered via Plotly. |
| **Interactive UI** | Click a point to preview, or run a slide-show; pop-up video player in upload list. |
| **BBox detector** | Optional YOLOv8 object-detector video overlay. |
| **Safe deletion** | CSRF-protected POST removes a video *and* every derived asset. |

---

## 🚀 Installation & Quick-start

### 1. Prerequisites

* **Python 3.10 or 3.11**  
* **ffmpeg** (for BBox video) — `brew install ffmpeg` or `apt install ffmpeg`

### 2. Set up

```bash
# clone & create a virtual environment
git clone https://github.com/<YOU>/Intelligent-Systems-Project.git
cd Intelligent-Systems-Project
python -m venv .venv && source .venv/bin/activate

# install requirements
pip install -r requirements.txt

# download YOLO weights (≈ 90 MB total)
python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n-pose.pt'); YOLO('yolov8n.pt')
PY

# initialise the database
python manage.py migrate

# run the dev server
python manage.py runserver
Browse to http://127.0.0.1:8000/ — you’re ready to upload a video.
```

## 🔗 Key AJAX / REST Endpoints
| Verb | Path                  | Params               | Returns                       |
| ---- | --------------------- | -------------------- | ----------------------------- |
| POST | `/upload/`            | `name`, `video_file` | `{"message": "..."}"`         |
| POST | `/deleteVideo/<vid>/` | —                    | Redirect → `/` (HTML) or JSON |
| GET  | `/generateData/`      | `video_ids=1,2`      | JSON status                   |
| GET  | `/make_distribution/` | `video_ids=…`        | JSON status                   |
| GET  | `/showDistribution/`  | `video_ids=…`        | Plotly figure JSON            |
| GET  | `/bboxDetect/`        | `video_id=`          | JSON `{ok}`                   |
| GET  | `/getVideo/`          | `path=`              | MP4 stream                    |
| GET  | `/getThumbList/`      | `video_id=`          | JSON list of thumbnails       |
| GET  | `/getImage/`          | `path=`              | JPEG                          |


## 🐞 Troubleshooting
| Issue                         | Remedy                                                        |
| ----------------------------- | ------------------------------------------------------------- |
| **YOLO `torch.load` warning** | Harmless — upstream note only.                                |
| **`ffmpeg` missing**          | Install via package manager.                                  |
| **Upload fails > 100 MB**     | Raise `DATA_UPLOAD_MAX_MEMORY_SIZE` or front-end proxy limit. |
| **SSL cert error on macOS**   | Run `Install Certificates.command` bundled with Python.       |
