# Stage 1: Build environment with compiler tools for heavy scientific packages
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies required for netCDF4 / numpy compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into a local user directory
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Lean production runtime image
FROM python:3.12-slim

WORKDIR /app

# Install runtime libraries required for NetCDF4/HDF5 dynamic linking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-103-1 \
    libnetcdf19 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy project source code
COPY . /app

# Default command to run the simulation pipeline
CMD ["python", "radiometry.py"]