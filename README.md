# Player Re-Identification in Football Broadcasts

This project implements real-time object detection and visual re-identification to ensure consistent player tracking across frames—even after occlusions or re-entry into the camera view.

---

## 🎥 Demo Videos

- ▶️ **Detection Output**:
  [![Detection Video](https://img.youtube.com/vi/aa5JK6-TQoI/maxresdefault.jpg)](https://youtu.be/aa5JK6-TQoI)
  [Watch Detected Video](https://drive.google.com/file/d/1deGHYQ-Ajzrs7ixgt3QeVnO2uEIHkuiw/view?usp=share_link)

- 🎯 **Tracking with Re-ID Output**:
  [![Tracking Video](https://img.youtube.com/vi/Fc4SgEace-Q/maxresdefault.jpg)](https://youtu.be/Fc4SgEace-Q)
  [Watch Tracked Video](https://drive.google.com/file/d/1deGHYQ-Ajzrs7ixgt3QeVnO2uEIHkuiw/view?usp=share_link)

---

## 📁 Folder Structure
```
project/
├── assets/
│   └── 15sec_input_720p.mp4
|   ├── best.pt
├── output/
│   ├── detected_frames/
│   ├── detected_video_custom.mp4
│   ├── tracked_frames/
│   └── tracked_video_reid_final.mp4
├── detect.py
├── track_with_global_reid.py
└── README.md
```

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/ankitk75/football-player-reid.git
cd player-reid
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
**Or install manually:**
```bash
pip install opencv-python torch torchvision numpy pillow scikit-learn ultralytics torchreid
```

### 4. Download YOLOv8 model weights
Place the `best.pt` file inside the `assets/` or project root directory. This should be your fine-tuned YOLOv8 model for player, referee, goalkeeper, and ball detection.

---

## ▶️ Running the Code

### 1. Basic Detection
```bash
python detect.py
```
- Outputs: `output/detected_frames/` (annotated images) and `detected_video_custom.mp4`

### 2. Tracking with Global Re-ID
```bash
python track_with_global_reid.py
```
- Outputs: `output/tracked_frames/` and `tracked_video_reid_final.mp4`

---

## 🛠 Dependencies
- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- Torch + TorchVision
- Torchreid
- scikit-learn
- Pillow

---

## ✅ Notes
- Code assumes CUDA GPU availability for Re-ID acceleration. Will fallback to CPU if not available.
- Re-ID consistency is based on cosine similarity between player feature embeddings.
- Detection confidence threshold is set at `0.4`.
