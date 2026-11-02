import numpy as np
from scipy.integrate import quad
import netCDF4 as nc
import matplotlib.pyplot as plt

# Physical Constants
H = 6.62607015e-34  # Planck's constant (J s)
C = 299792458.0     # Speed of light (m/s)
K_B = 1.380649e-23  # Boltzmann's constant (J/K)

def planck_wavelength(wavelength_m, temperature_k):
    """Computes spectral radiance of a blackbody per meter wavelength."""
    exponent = (H * C) / (wavelength_m * K_B * temperature_k)
    return (2.0 * H * C**2 / (wavelength_m**5)) / (np.exp(exponent) - 1.0)

def integrate_mwir_band(temp_k, emiss=1.0, lam_min=3.5e-6, lam_max=4.1e-6):
    """Integrates Planck's law over the MWIR spectral band (3.5 - 4.1 um)."""
    radiance, _ = quad(planck_wavelength, lam_min, lam_max, args=(temp_k,))
    return emiss * radiance

def simulate_fpa_frame(shape=(128, 128), scene_temp=300.0, t_int=0.01, dark_current=100.0, read_noise_sigma=5.0, gain_variation=0.05):
    """
    Simulates a staring FPA frame with photon shot noise, dark current, read noise,
    and pixel-to-pixel responsivity variations (Fixed Pattern Noise).
    """
    base_radiance = integrate_mwir_band(scene_temp)
    mean_electrons = base_radiance * t_int * 1e-5
    
    # Introduce pixel-to-pixel gain variation (Fixed Pattern Noise simulation)
    pixel_gains = 1.0 + np.random.normal(0.0, gain_variation, size=shape)
    
    shot_noise_electrons = np.random.poisson(mean_electrons * pixel_gains, size=shape)
    dark_electrons = np.random.poisson(dark_current * t_int, size=shape)
    read_noise = np.random.normal(0.0, read_noise_sigma, size=shape)
    
    total_signal = shot_noise_electrons + dark_electrons + read_noise
    return np.clip(total_signal, 0, 65535)

def compute_two_point_nuc(frame_cold_measured, frame_hot_measured, known_cold_rad, known_hot_rad):
    """
    Computes Two-Point Non-Uniformity Correction (NUC) gain and offset matrices.
    Corrects for pixel-to-pixel gain and offset variations across the FPA.
    """
    # Desired uniform response across pixels based on known standard radiance inputs
    # Gain matrix: g_ij = (R_hot - R_cold) / (S_hot_ij - S_cold_ij)
    # Offset matrix: o_ij = R_cold - g_ij * S_cold_ij
    signal_diff = frame_hot_measured - frame_cold_measured
    # Avoid division by zero
    signal_diff = np.where(signal_diff == 0, 1e-6, signal_diff)
    
    gain_matrix = (known_hot_rad - known_cold_rad) / signal_diff
    offset_matrix = known_cold_rad - (gain_matrix * frame_cold_measured)
    
    return gain_matrix, offset_matrix

def apply_nuc(raw_frame, gain_matrix, offset_matrix):
    """Applies NUC gain and offset calibration to a raw frame."""
    return (raw_frame * gain_matrix) + offset_matrix

def save_to_netcdf(filename, raw_frame, calibrated_frame, gain_matrix, offset_matrix, temp_scene):
    """Saves multi-dimensional FPA data and metadata into a professional NetCDF4 file."""
    with nc.Dataset(filename, 'w', format='NETCDF4') as ds:
        # Define dimensions
        ds.createDimension('height', raw_frame.shape[0])
        ds.createDimension('width', raw_frame.shape[1])
        ds.createDimension('time', 1)  # Single snapshot frame for now
        
        # Global Attributes (Metadata)
        ds.description = "Simulated OPIR MWIR FPA Telemetry & Calibration Data"
        ds.spectral_band = "3.5 - 4.1 um"
        ds.orbit_regime = "GEO (~35,786 km)"
        
        # Create Variables
        raw_var = ds.createVariable('raw_dn', 'f4', ('time', 'height', 'width'))
        cal_var = ds.createVariable('calibrated_radiance', 'f4', ('time', 'height', 'width'))
        gain_var = ds.createVariable('nuc_gain', 'f4', ('height', 'width'))
        offset_var = ds.createVariable('nuc_offset', 'f4', ('height', 'width'))
        
        # Populate Data
        raw_var[0, :, :] = raw_frame
        cal_var[0, :, :] = calibrated_frame
        gain_var[:, :] = gain_matrix
        offset_var[:, :] = offset_matrix
        
        # Variable Attributes
        raw_var.units = "Digital Numbers (DN)"
        cal_var.units = "W / (m^2 * sr)"
        
    print(f"Successfully exported NetCDF4 data cube to {filename}")

def plot_radiometry_results(wavelengths, spectral_radiances, raw_frame, calibrated_frame, gain_matrix):
    """Generates professional diagnostic plots for MWIR Planck curves and NUC correction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Planck's Law Curve
    axes[0].plot(wavelengths * 1e6, spectral_radiances, color='purple', lw=2)
    axes[0].set_title("MWIR Blackbody Spectrum (850K)")
    axes[0].set_xlabel("Wavelength ($\mu m$)")
    axes[0].set_ylabel("Spectral Radiance ($W/m^2/sr/\mu m$)")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Raw FPA Frame (with Fixed Pattern Noise & Noise)
    im1 = axes[1].imshow(raw_frame, cmap='inferno')
    axes[1].set_title("Raw FPA Frame (with FPN & Noise)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. NUC Calibrated Radiance Frame
    im2 = axes[2].imshow(calibrated_frame, cmap='viridis')
    axes[2].set_title("NUC-Calibrated Radiance Frame")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("radiometry_nuc_performance.png", dpi=300)
    print("Successfully generated and saved radiometry figure: radiometry_nuc_performance.png")
    plt.show()

if __name__ == "__main__":
    print("--- Running OPIR Radiometry & NUC Pipeline ---")
    
    # 1. Simulate Calibration Blackbodies (Cold target @ 280K, Hot target @ 350K)
    rad_cold = integrate_mwir_band(280.0)
    rad_hot = integrate_mwir_band(350.0)
    
    frame_cold = simulate_fpa_frame(scene_temp=280.0)
    frame_hot = simulate_fpa_frame(scene_temp=350.0)
    
    # 2. Compute NUC matrices
    gain_mat, offset_mat = compute_two_point_nuc(frame_cold, frame_hot, rad_cold, rad_hot)
    print(f"NUC Matrices Computed. Gain Shape: {gain_mat.shape}, Offset Shape: {offset_mat.shape}")
    
    # 3. Simulate an operational scene (e.g., Target Plume @ 850K)
    scene_temp = 850.0
    raw_scene_frame = simulate_fpa_frame(scene_temp=scene_temp)
    
    # 4. Apply Calibration
    calibrated_frame = apply_nuc(raw_scene_frame, gain_mat, offset_mat)
    print(f"Calibrated Scene Radiance - Min: {calibrated_frame.min():.4e}, Max: {calibrated_frame.max():.4e}")
    
    # 5. Export to NetCDF4
    save_to_netcdf("opir_module1_output.nc", raw_scene_frame, calibrated_frame, gain_mat, offset_mat, scene_temp)

    # Generate visualization data
    wl_range = np.linspace(3.5e-6, 4.1e-6, 100)
    spec_rads = [planck_wavelength(wl, scene_temp) for wl in wl_range]
    plot_radiometry_results(wl_range, np.array(spec_rads), raw_scene_frame, calibrated_frame, gain_mat)