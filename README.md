# 🏃‍♀️ IPRO – no06 Video-Analytics Demo

A lightweight Django demo that lets you

1. **Upload videos**
2. **Extract pose-based RGB features** (4 keypoints × RGB = 12-D)
3. **Cluster them** with TSNE (3-D) + K-Means (5 clusters)
4. **Visualize** everything in an interactive 3-D scatter plot

![screenshot](docs/infos.png)

---

## ✨ Features

| Button                | Action |
|-----------------------|--------|
| **Upload Video**      | Save an MP4 file to `media/videos/` |
| **Generate Data**     | Run YOLOv8-Pose and save RGB feature vectors |
| **Make Distribution** | TSNE → K-Means and write results back to DB |
| **Show Distribution** | Plotly 3-D scatter + thumbnail on click |
| **File List**         | Quick source-file browser |
| **Extra Features**    | Switch to red-header *extra* screen (includes One-Click mode) |

---

## 🛠️ Quick Start (Local)

```bash
# 1) virtualenv (recommended)
python -m venv venv
source venv/bin/activate

# 2) dependencies
pip install -r requirements.txt   # ultralytics, django==5.2, etc.

# 3) migrate DB
python manage.py migrate

# 4) dev server
python manage.py runserver
