# syntax=docker/dockerfile:1.7
ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

# Base build deps. We pull cmake from Kitware's APT repo so we get a current
# version (Ubuntu 22.04's default cmake is 3.22 which is below our minimum).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git gnupg lsb-release \
      ninja-build pkg-config python3 wget vim \
 && wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/kitware.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends cmake \
 && rm -rf /var/lib/apt/lists/*

# Build the project from the host's working tree (NOT a fresh git clone).
# Mount your source on /workspace via docker-compose, or `docker run -v ...`.
WORKDIR /workspace

# Optional one-shot build: `docker build --target builder` will compile the
# source baked into the image. The default `runtime` stage assumes you bind-
# mount your source and build interactively.
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder
COPY --from=0 / /
COPY . /workspace
WORKDIR /workspace
RUN cmake --preset release-multi \
 && cmake --build --preset release-multi -j

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS runtime
COPY --from=0 / /
WORKDIR /workspace
CMD ["/bin/bash"]
