# A PyTorch client with no GPU and no NVIDIA driver.
#
# Stock PyTorch and its stock CUDA libraries. The only thing replaced is
# libcuda.so.1, which is our shim: LD_LIBRARY_PATH puts it ahead of anything
# else, and since a driverless machine has no real one, nothing competes.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# aarch64 CUDA builds of PyTorch live on the pytorch index, not PyPI. The
# nvidia-* CUDA runtime and math libraries come along as dependencies and run
# unmodified on the client; only the driver call underneath them is remoted.
ARG TORCH_VERSION=2.9.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"

# The shim, built by scripts/build_client.sh.
COPY libcuda.so.1 /opt/rgpu/lib/libcuda.so.1
RUN ln -s libcuda.so.1 /opt/rgpu/lib/libcuda.so

ENV LD_LIBRARY_PATH=/opt/rgpu/lib
ENV RGPU_SERVER=host.docker.internal:9713

WORKDIR /work
CMD ["python3", "-c", "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available())"]
