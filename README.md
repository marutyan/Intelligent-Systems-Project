# Intelligent-Systems-Project / **no06**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%20|%203.10-blue)
![Django](https://img.shields.io/badge/django-5.2-green)
![ultralytics-yolo](https://img.shields.io/badge/YOLOv8-pose%20%26%20det-red)

> **no06** is a lightweight Django web app that turns raw classroom videos into **interactive 3-D distributions** of pose-derived colour features, lets you **cluster** frames with *t-SNE + k-means* on the fly, and review results through slick, single-page controls.

<div align="center">

![demo gif](docs/demo.gif)
<sub><i>Replace with a short screen-capture showing upload → clustering → pop-up playback.</i></sub>
</div>

---

## 1 · Features

| Category | Highlights |
|----------|------------|
| **Upload & storage** | Drag-and-drop or classic form (AJAX), videos persisted under `no06/media/videos/` with metadata in SQLite (`Video` model). |
| **Frame sampling** | 1 fps extraction, pose detection on 4 joints (both shoulders & hips) via **YOLOv8-pose**, then RGB averaging → 12-dim feature vector. |
| **Clustering & visualisation** | t-SNE ↦ 3-D scatter + **k-means (k = 5)**; colour palette from Plotly Set2. |
| **Preview & slide-show** | Click any point to view its thumbnail, or launch an auto-cycling slide-show of all frames for that video. |
| **BBox detector** | Optional YOLOv8 object detector to render bounding-box videos (fast-start H.264). |
| **File browser (debug)** | One-click directory listing of every source/template/script for fast pinpointing during class. |
| **Admin-free deletion** | CSRF-protected POST endpoint deletes a video **and** all derived artefacts, with UI confirmation. |
| **Responsive UI** | Pure vanilla JS + CSS variables; looks neat on 1440-px desktop down to 13-inch notebook size. |

---

## 2 · Quick-start

### 2.1 Docker (one-liner)

```bash
docker compose up --build
# visit http://localhost:8000/
2.2 Bare-metal (macOS / Linux)
bash
コピーする
git clone https://github.com/<YOU>/Intelligent-Systems-Project.git
cd Intelligent-Systems-Project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# pull YOLO weights (≈ 6 MB + 80 MB)
python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8n-pose.pt'), YOLO('yolov8n.pt')
PY

python manage.py migrate
python manage.py runserver
GPU: If CUDA is available, install torch==2.* with the appropriate +cu* extra for faster inference.

3 · Environment variables
Name	Default	Purpose
IPRO_APP_NAME	no06	Select which Django app is mounted at root.
YOLO_POSE_WEIGHTS	yolov8n-pose.pt	Override to use larger yolov8m etc.
YOLO_DET_WEIGHTS	yolov8n.pt	Object-detector weights.
MEDIA_ROOT	<BASE>/no06/media	Central storage of uploads & derivatives.

4 · REST / AJAX endpoints
Verb	Path	Query / Body	Returns
POST	/upload/	name, video_file	JSON {message}
POST	/deleteVideo/<vid>/	–	Redirect ▶ / (HTML) or JSON
GET	/generateData/	video_ids=1,2	JSON message
GET	/make_distribution/	video_ids=…	JSON message
GET	/showDistribution/	video_ids=…	Plotly figure JSON
GET	/bboxDetect/	video_id=	JSON {ok}
GET	/getVideo/	path=	video/mp4 stream
GET	/getThumbList/	video_id=	JSON list of thumbnails
GET	/getImage/	path=	image/jpeg

Example: delete a video
bash
コピーする
curl -X POST http://localhost:8000/deleteVideo/3/ \
     -H 'X-Requested-With: XMLHttpRequest'
5 · Development workflow
bash
コピーする
pre-commit install            # black, isort, flake8, djhtml, etc.
python manage.py test
black --check . && ruff check . && mypy no06/
6 · Troubleshooting
Symptom	Fix
YOLO raises torch.load warnings	They’re safe; see upstream notice.
ffmpeg not found	brew install ffmpeg (mac) or apt install ffmpeg.
Videos > 100 MB fail	Increase DATA_UPLOAD_MAX_MEMORY_SIZE or use Nginx’s client_max_body_size.
SSL errors on macOS	Run /Applications/Python\ 3.x/Install\ Certificates.command.

7 · Road-map
✅ Per-video delete (cascade physical files & DB)

🔄 Refactor frontend into Vue 3

⏩ Batch feature extraction using Celery + Redis

📈 Export cluster assignments as CSV/Parquet

🧪 Raise test coverage to 90 %

8 · Contributing
Fork → create feature branch (git checkout -b feat/awesome).

Add or update tests.

Run pre-commit run --all-files.

Push & open a PR against main.

Explain why the change benefits students / researchers.

9 · License
Released under the MIT License – see LICENSE for details.

Made with ☕️ & YOLO v8 in Osaka, 2025-06.