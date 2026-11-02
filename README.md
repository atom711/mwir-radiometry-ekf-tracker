# OPIR Simulated Sensor & Processing Pipeline

## Overview
A production-grade, simulated Over-the-Top Persistent Infrared (OPIR) data processing pipeline built for Geostationary Earth Orbit (GEO) staring sensor architectures. This repository demonstrates core technical proficiencies in MWIR radiometry, focal plane array (FPA) noise modeling, Two-Point Non-Uniformity Correction (NUC), hierarchical data serialization (NetCDF4), and 3D kinematics/Extended Kalman Filtering.

---

## Architecture & Modules

### Module 1: Radiometric & Spectral Response Physics
* **Planck Integration:** Numerical integration (`scipy.integrate.quad`) of blackbody spectral radiance across the MWIR band ($3.5 - 4.1\ \mu m$).
* **FPA Noise Modeling:** Root-sum-square (RSS) noise budget combining Poisson-distributed photon shot noise, Poisson dark current noise, and additive Gaussian Readout Integrated Circuit (ROIC) read noise.
* **Two-Point NUC:** Eliminates Fixed Pattern Noise (FPN) by calculating per-pixel gain and offset correction matrices using cold and hot reference calibration frames.
* **Hierarchical I/O:** Serializes multi-dimensional FPA time-series, calibrated radiance frames, and calibration matrices into structured NetCDF4 data cubes.