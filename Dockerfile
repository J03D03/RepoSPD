FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    python3-pip \
    python3-dev \
    python-is-python3 \
    cflow \
    openjdk-8-jre \
    maven \
    clang \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/lib/x86_64-linux-gnu/libclang-*.so.1 /usr/lib/x86_64-linux-gnu/libclang.so 2>/dev/null || true

RUN pip install --upgrade pip
RUN pip install numpy==1.24.4 pandas jsonlines tree-sitter==0.21.1 transformers==4.41.2 clang==6.0.0.2

# Install PyTorch 2.1.0 with CUDA 12.1 support for H200 (sm_90)
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric dependencies for Torch 2.1.0 + CUDA 12.1
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
RUN pip install torch-geometric==2.4.0

# Create non-root user with host UID/GID
ARG UID=1000
ARG GID=100
RUN groupadd -g $GID -o user 2>/dev/null || true && \
    useradd -m -u $UID -g $GID -o -s /bin/bash user

# Create the working directory
RUN mkdir -p /RepoSPD && chown user:$GID /RepoSPD
WORKDIR /RepoSPD

USER user
