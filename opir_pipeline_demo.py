import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, label

# ==========================================
# STEP 1: Generate a Synthetic HDF5 Data Cube
# ==========================================
file_path = "synthetic_opir_frame.h5"

print("Generating synthetic HDF5 OPIR data cube...")
# Simulate a time series of focal plane array (FPA) frames: 10 frames, 256x256 pixels
num_frames, height, width = 10, 256, 256

# Create random background noise (thermal/sensor noise)
background_noise = np.random.normal(loc=100.0, scale=5.0, size=(num_frames, height, width))

# Inject a moving target (e.g., a boosting missile plume starting at coordinate [50, 100] moving diagonally)
target_cube = background_noise.copy()
for t in range(num_frames):
    pos_y = 50 + (t * 3)
    pos_x = 100 + (t * 2)
    # Draw a bright Gaussian blob representing the infrared signature
    y_grid, x_grid = np.ogrid[:height, :width]
    target_blob = 250.0 * np.exp(-((y_grid - pos_y)**2 + (x_grid - pos_x)**2) / (2 * 3.0**2))
    target_cube[t] += target_blob

# Write to HDF5 file
with h5py.File(file_path, "w") as f:
    dset = f.create_dataset("fpa_frames", data=target_cube, compression="gzip")
    f.attrs["sensor_name"] = "SIMULATED_MWIR_FPA"
    f.attrs["wavelength_microns"] = 4.2  # Typical CO2 absorption/plume band
    f.attrs["units"] = "Watts/steradian/m^2"

print(f"Successfully wrote synthetic file to: {file_path}\n")

# ==========================================
# STEP 2: Read and Process the HDF5 Data
# ==========================================
print("Reading HDF5 file and executing processing pipeline...")

with h5py.File(file_path, "r") as f:
    # Inspect metadata
    print(f"-> Reading Dataset Attributes:")
    for key, val in f.attrs.items():
        print(f"   {key}: {val}")
        
    # Load data array into memory (or slice it)
    data_cube = f["fpa_frames"][:]
    print(f"-> Loaded data cube shape: {data_cube.shape}")

# Image Processing Pipeline:
processed_frames = []
for t in range(data_cube.shape.isize if hasattr(data_cube.shape, 'isize') else data_cube.shape[0]):
    frame = data_cube[t]
    
    # 1. Spatial Filtering / Noise Reduction (Gaussian blur to smooth high-frequency FPA noise)
    smoothed = gaussian_filter(frame, sigma=1.0)
    
    # 2. Non-Uniformity / Background Subtraction (Temporal or spatial background removal)
    # Estimate background level via median and subtract
    background_level = np.median(smoothed)
    cleaned = smoothed - background_level
    cleaned[cleaned < 0] = 0  # Clip negative values
    
    # 3. Thresholding & Feature Extraction (Find bright blobs above noise floor)
    threshold = 20.0  # Detection threshold
    binary_mask = cleaned > threshold
    
    # Connected component labeling to isolate potential targets
    labeled_array, num_features = label(binary_mask)
    
    processed_frames.append({
        "frame_index": t,
        "detections_found": num_features,
        "max_intensity": np.max(cleaned)
    })

print("\n--- Pipeline Processing Summary ---")
for res in processed_frames:
    print(f"Frame {res['frame_index']:02d}: Detected {res['detections_found']} target object(s) | Peak Intensity: {res['max_intensity']:.2f}")