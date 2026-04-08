# 🧭 Vision Pipeline – Visual Odometry Module

## 🎯 Objective

The goal of this module is to estimate the **relative motion of the drone** using consecutive image frames.

It provides the foundation for **GPS-denied navigation** by answering:

* How much did the drone move?
* In which direction?
* How did it rotate?

---

## 🧠 Core Concept

This module implements a **classical visual odometry pipeline** based on feature tracking and geometric motion estimation.

Pipeline:

```
Frame_t → Frame_t+1
    ↓
Feature Detection (ORB)
    ↓
Feature Matching
    ↓
Essential Matrix Estimation
    ↓
Pose Recovery (R, t)
```

---

## 🚀 Key Responsibilities

* Detect visual features in each frame
* Match features between consecutive frames
* Estimate relative camera motion
* Output rotation and translation vectors

---

## 📥 Inputs

* Sequential image frames:

```
data/frames/
  frame_000000.png
  frame_000001.png
  ...
```

* Camera intrinsic matrix (K)

---

## 📤 Outputs

### 1. Relative Motion

For each frame pair:

* **Rotation (R)** → 3×3 matrix
* **Translation (t)** → 3×1 vector

Example:

```
Translation: [0.01, -0.02, 0.98]

Rotation:
[[...],
 [...],
 [...]]
```

---

### 2. (Optional Future)

* Estimated trajectory
* Visualization plots
* Drift analysis

---

## 🛠️ Technologies Used

* OpenCV:

  * ORB feature detector
  * BFMatcher
  * Essential matrix estimation
  * Pose recovery

* NumPy:

  * Matrix operations
  * Coordinate handling

---

## ⚙️ Design Highlights

* **Lightweight and real-time capable**
* Fully **modular architecture**
* Easy to extend with:

  * Deep Learning models (DeepVO)
  * IMU fusion (VIO)
  * SLAM systems

---

## 🔬 Algorithm Details

### Feature Detection

* ORB (Oriented FAST and Rotated BRIEF)
* Efficient and robust for real-time applications

### Feature Matching

* Brute-force Hamming distance
* Cross-check enabled for reliability

### Motion Estimation

* Essential matrix via RANSAC
* Robust to outliers

### Pose Recovery

* Extract rotation and translation between frames

---

## ⚠️ Limitations

* Scale ambiguity (monocular setup)
* Accumulated drift over time
* Sensitive to:

  * Motion blur
  * Low-texture environments

---

## 🧪 Usage Example

```bash
python3 run_odometry.py
```

---

## 🔗 Role in the Full System

This module is the **core estimation engine** and feeds into:

➡️ Visual-Inertial Odometry (VIO)
➡️ Flight controller (PX4)
➡️ Navigation & control modules

---

## 🚀 Future Extensions

* IMU fusion (VIO)
* Deep learning-based motion estimation
* Loop closure & SLAM
* Trajectory optimization

---

## ✅ Expected Outcome

A working system that:

* Tracks motion between frames
* Provides relative pose estimates

This is the **first step toward full autonomous navigation without GPS**.

---
