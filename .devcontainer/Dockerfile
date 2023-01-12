FROM nvidia/cuda:11.3.0-cudnn8-devel-ubi8

ENV PATH /root/miniconda3/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    cmake \
    curl \
    htop \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    wget \
    unzip \
    bzip2 \
    build-essential \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ../environment-dev.yml environment.yml
RUN curl "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o Miniconda_Install.sh && \
    bash Miniconda_Install.sh -b && \
    rm Miniconda_Install.sh && \
    bash -c "source /root/miniconda3/bin/activate && conda install mamba -c conda-forge && conda init && mamba env update"

RUN pip install nvidia-pyindex && \
    pip install tritonclient[http,grpc]

# Install zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -p git \
    # -p ssh-agent \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting

RUN conda init zsh

WORKDIR /workspace
