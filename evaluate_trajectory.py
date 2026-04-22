"""
Trajectory Evaluation (EuRoC)

This script compares estimated trajectory with ground truth.

Steps:
- Load estimated trajectory
- Load ground truth
- Align (scale + translation)
- Compute error (RMSE)
- Plot comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ===============================
# Paths
# ===============================
EST_PATH = "trajectory.npy"
GT_PATH = "MH_01_easy/mav0/state_groundtruth_estimate0/data.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Load estimated trajectory
# ===============================
est = np.load(EST_PATH)

# ===============================
# Load ground truth
# ===============================
gt_data = pd.read_csv(GT_PATH)

gt = gt_data[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].values

# ===============================
# Match lengths
# ===============================
min_len = min(len(est), len(gt))
est = est[:min_len]
gt = gt[:min_len]

# ===============================
# Scale alignment (very important)
# ===============================
scale = np.linalg.norm(gt) / np.linalg.norm(est)
est_scaled = est * scale

# ===============================
# Translation alignment
# ===============================
offset = gt[0] - est_scaled[0]
est_aligned = est_scaled + offset

# ===============================
# RMSE Error
# ===============================
error = est_aligned - gt
rmse = np.sqrt(np.mean(np.sum(error**2, axis=1)))

print(f"✅ RMSE Error: {rmse:.4f} meters")

# ===============================
# Plot comparison (Top View)
# ===============================
plt.figure()
plt.plot(gt[:, 0], gt[:, 2], label="Ground Truth")
plt.plot(est_aligned[:, 0], est_aligned[:, 2], label="Estimated")

plt.legend()
plt.title("Trajectory Comparison (Top View)")
plt.xlabel("X")
plt.ylabel("Z")
plt.axis("equal")
plt.grid()

plt.savefig(os.path.join(OUTPUT_DIR, "comparison_top_view.png"))
plt.close()

# ===============================
# 3D Plot
# ===============================
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], label="GT")
ax.plot(est_aligned[:, 0], est_aligned[:, 1], est_aligned[:, 2], label="Estimated")

ax.legend()
ax.set_title("3D Trajectory Comparison")

plt.savefig(os.path.join(OUTPUT_DIR, "comparison_3d.png"))
plt.close()

print(f"📊 Plots saved in {OUTPUT_DIR}")
