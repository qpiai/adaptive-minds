FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY build/requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy the project files
COPY . .

# Create a script to run the inference (but don't run it by default)
RUN echo '#!/bin/bash\ncd /app/build\npython test_docker.py' > /app/run_inference.sh && \
    chmod +x /app/run_inference.sh

# Expose port if needed for web interface
EXPOSE 8000

# Default command - just start bash for interactive use
CMD ["/bin/bash"] 