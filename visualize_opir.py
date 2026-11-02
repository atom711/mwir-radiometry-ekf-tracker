import h5py
import matplotlib.pyplot as plt
import numpy as np

# 1. Open your HDF5 file
file_path = "synthetic_opir_frame.h5"  # Replace with your actual file path

with h5py.File(file_path, "r") as f:
    # Print available keys/datasets in the file to confirm structure
    print("Datasets in HDF5 file:", list(f.keys()))
    
    # Load the data cube into memory
    # (Adjust dataset name 'fpa_frames' to match whatever key is inside your specific file)
    data_cube = f["fpa_frames"][:]

print(f"Data cube shape: {data_cube.shape} (Frames, Height, Width)")

# ==========================================
# Option A: Plot a Single Frame
# ==========================================
frame_to_view = 0  # Change this to look at different time steps
plt.figure(figsize=(6, 6))
plt.imshow(data_cube[frame_to_view], cmap="inferno", origin="lower")
plt.colorbar(label="Radiance / Intensity")
plt.title(f"OPIR FPA Frame {frame_to_view}")
plt.xlabel("X Pixel")
plt.ylabel("Y Pixel")
plt.show()

# ==========================================
# Option B: Plot a Multi-Frame Grid Sequence
# ==========================================
# Great for seeing a target move across time steps in a static view
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for t in range(min(10, data_cube.shape[0])):
    ax = axes[t]
    im = ax.imshow(data_cube[t], cmap="inferno", origin="lower")
    ax.set_title(f"T = {t}")
    ax.axis("off")

plt.suptitle("Sequential OPIR Focal Plane Array Frames", fontsize=14)
plt.tight_layout()
plt.show()