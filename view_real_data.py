import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==========================================
# 1. Load Your Real HDF5 File
# ==========================================
file_path = "GRIDSAT-B1.2026.09.30.21.v02r01.nc"
dataset_name = "irwin_cdr"  # <--- Change this to the dataset key inside your file

print(f"Opening file: {file_path}")
with h5py.File(file_path, "r") as f:
  print("Available top-level keys in file:", list(f.keys()))

  # Load the multi-dimensional array into memory
  # (If it's deeply nested, e.g., f['Group']['Subgroup']['dataset'], adjust accordingly)
  data_cube = f[dataset_name][:]

print(f"Data cube shape loaded: {data_cube.shape}")

# Ensure we are working with a 3D array (Frames, Height, Width)
# If your dataset has more or fewer dimensions, you may need to slice it (e.g., data_cube[:, 0, :, :])
if len(data_cube.shape) == 4:
  print(
      "Detected 4D array (e.g., [Time, Band, Height, Width]). Selecting first"
      " band..."
  )
  data_cube = data_cube[:, 0, :, :]  # Select first spectral band, for example

# ==========================================
# 2. Build the Interactive Viewer with Slider
# ==========================================
fig, ax = plt.subplots(figsize=(8, 8))
# Adjust bottom margin to make room for the slider control panel
plt.subplots_adjust(bottom=0.25)

initial_frame = 0
# Render the first frame
im = ax.imshow(data_cube[initial_frame], cmap="inferno", origin="lower")
ax.set_title(
    f"File: {file_path} | Frame {initial_frame}/{data_cube.shape[0]-1}"
)
cbar = plt.colorbar(im, ax=ax, label="Sensor DN / Radiance")

# Define position for the slider [left, bottom, width, height]
ax_slider = plt.axes([0.20, 0.1, 0.65, 0.03])
frame_slider = Slider(
    ax=ax_slider,
    label="Frame Index",
    valmin=0,
    valmax=data_cube.shape[0] - 1,
    valinit=initial_frame,
    valfmt="%d",
)


# Update function triggered when you drag the slider
valmin, valmax = 0, data_cube.shape[0] - 1


def update(val):
  frame_idx = int(frame_slider.val)
  im.set_data(data_cube[frame_idx])

  # Auto-adjust color scaling per frame if contrast changes significantly,
  # or comment this out to keep a fixed global color scale:
  im.set_clim(vmin=np.min(data_cube[frame_idx]), vmax=np.max(data_cube[frame_idx]))

  ax.set_title(f"File: {file_path} | Frame {frame_idx}/{valmax}")
  fig.canvas.draw_idle()


# Link the slider to the update function
frame_slider.on_changed(update)

plt.show()