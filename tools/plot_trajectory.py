"""
Trajectory Visualization Tool

This script loads a saved trajectory (trajectory.npy) and generates
multiple plots for analysis:

- Top view (X-Z)
- Side view (X-Y)
- 3D trajectory
- Axis-wise position over time

All plots are saved in the "outputs" directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Config
# ===============================
TRAJ_PATH = "trajectory.npy"
OUTPUT_DIR = "outputs"

# ===============================
# Create output directory
# ===============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Load trajectory
# ===============================
traj = np.load(TRAJ_PATH)

x = traj[:, 0]
y = traj[:, 1]
z = traj[:, 2]

# ===============================
# 1. Top View (X-Z)
# ===============================
plt.figure()
plt.plot(x, z)
plt.title("Top View (X-Z)")
plt.xlabel("X")
plt.ylabel("Z")
plt.axis("equal")
plt.grid()

plt.savefig(os.path.join(OUTPUT_DIR, "top_view.png"))
plt.close()

# ===============================
# 2. Side View (X-Y)
# ===============================
plt.figure()
plt.plot(x, y)
plt.title("Side View (X-Y)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid()

plt.savefig(os.path.join(OUTPUT_DIR, "side_view.png"))
plt.close()

# ===============================
# 3. 3D Trajectory
# ===============================
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(x, y, z)
ax.set_title("3D Trajectory")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_3d.png"))
plt.close()

# ===============================
# 4. Position over time
# ===============================
t = np.arange(len(x))

plt.figure()
plt.plot(t, x, label="X")
plt.plot(t, y, label="Y")
plt.plot(t, z, label="Z")

plt.title("Position over Time")
plt.xlabel("Frame")
plt.ylabel("Position")
plt.legend()
plt.grid()

plt.savefig(os.path.join(OUTPUT_DIR, "position_over_time.png"))
plt.close()

# ===============================
# Done
# ===============================
print(f"✅ Plots saved in: {OUTPUT_DIR}")
