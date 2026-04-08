# 📸 Vision Pipeline – Frame Capture Module

## 🎯 Objective

The goal of this module is to **reliably capture time-synchronized image frames** from various video sources and prepare structured data for downstream visual odometry and state estimation.

This module serves as the **data acquisition layer** of the GPS-denied navigation pipeline.

---

## 🚀 Key Responsibilities

* Capture image frames from multiple sources:

  * Webcam
  * Video files
  * RTSP / UDP streams
  * Simulator feeds (PX4 SITL / Gazebo)

* Maintain a **fixed and controllable frame rate**

* Generate **precise timestamps** for each frame

* Store frames and metadata in a structured format

---

## 📥 Inputs

* Video source:

  * Device index (`0`, `1`, ...)
  * File path (`.mp4`, `.avi`, ...)
  * Network stream (`rtsp://`, `udp://`)

* Configuration parameters:

  * Target FPS
  * Output directory

---

## 📤 Outputs

### 1. Image Frames

Stored as sequential files:

```
frames/
  frame_000000.png
  frame_000001.png
  ...
```

---

### 2. Metadata Log (CSV)

Each frame is logged with timing information:

```
logs/capture_log_*.csv
```

| field               | description              |
| ------------------- | ------------------------ |
| frame_id            | Sequential frame index   |
| system_time_sec     | Unix timestamp (float)   |
| image_timestamp_iso | ISO-8601 timestamp (UTC) |
| file_name           | Corresponding image file |

---

## 🛠️ Technologies Used

* OpenCV (`cv2`) → video capture & image writing
* Python standard libraries:

  * `time` → timing control
  * `datetime` → timestamp generation
  * `csv` → structured logging
  * `pathlib` → file management

---

## ⚙️ Design Highlights

* **Multi-source compatibility** (webcam, file, stream)
* **Deterministic frame rate control**
* **High-precision timestamp logging**
* **Clean directory structure for downstream processing**

---

## 🧪 Usage Example

```bash
python3 capture_frames.py --source video.mp4 --fps 10 --output .
```

---

## 📌 Notes

* In WSL environments, direct webcam access may be limited.
* For initial testing, using a video file is recommended.
* This module is intentionally **decoupled** from downstream processing to ensure modularity.

---

## 🔗 Role in the Full System

This module feeds data into:

➡️ Visual Odometry (Motion Estimation)
➡️ Visual-Inertial Fusion (VIO)
➡️ Future Deep Learning modules

---

## ✅ Expected Outcome

A clean dataset of:

* Time-aligned frames
* Accurate timestamps

Ready for motion estimation and trajectory reconstruction.

---
