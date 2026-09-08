# Compiles the shim. Needs a C++ toolchain and cuda.h; notably it does NOT
# need the CUDA toolkit or the driver, because the shim replaces the driver
# rather than linking it.
FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
