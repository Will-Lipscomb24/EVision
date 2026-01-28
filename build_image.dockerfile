# Use the official Ubuntu base image
FROM ubuntu:24.04

# Avoid prompts from apt during build
ENV DEBIAN_FRONTEND=noninteractive

# Update and install core system utilities
RUN apt-get update && apt-get install -y \
    apt-utils \
    build-essential \
    software-properties-common \
    wget \
    unzip \
    curl \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install development libraries and dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libboost-all-dev \
    libusb-1.0-0-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install graphics, HDF5, and multimedia tools
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    hdf5-tools \
    libglew-dev \
    libglfw3-dev \
    libcanberra-gtk-module \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

COPY /home/will/projects/EVision/tools/openeb/utils/python/requirements_openeb.txt .
COPY /home/will/projects/EVision/tools/openeb/utils/python/requirements_pytorch_cuda.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_openeb.txt \
    pip install --no-cache-dir -r requirements_pytorch_cuda

# Default command
CMD ["/bin/bash"]