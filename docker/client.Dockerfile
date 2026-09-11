# A PyTorch client with no GPU and no NVIDIA driver.
#
# Stock PyTorch and stock CUDA math libraries. Two libraries are replaced:
# libcuda.so.1, which forwards driver calls to the GPU host, and
# libcudart.so.12, which translates the runtime API into driver calls. The
# stock runtime cannot be used, because it demands undocumented driver export
# tables that cannot cross a process boundary.
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.9.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# torch is installed without its dependency closure, but less can be left out
# than it looks: this build names nccl, nvshmem, cufile and cusparselt in
# DT_NEEDED, so import torch fails without them however little we intend to
# use them. Only triton is genuinely optional, and only until torch.compile.
# Versions are pinned to what this torch build resolves to.
#
# nvidia-cuda-runtime-cu12 is deliberately absent: our libcudart.so.12 stands
# in for it, and not installing the stock one removes any question of which
# gets loaded.
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX}" --no-deps \
      "torch==${TORCH_VERSION}" \
 && pip install --no-cache-dir --index-url "${TORCH_INDEX}" \
      --extra-index-url https://pypi.org/simple \
      filelock typing-extensions sympy networkx jinja2 fsspec \
      nvidia-cublas-cu12==12.8.4.1 \
      nvidia-cudnn-cu12==9.10.2.21 \
      nvidia-cufft-cu12==11.3.3.83 \
      nvidia-curand-cu12==10.3.9.90 \
      nvidia-cusolver-cu12==11.7.3.90 \
      nvidia-cusparse-cu12==12.5.8.93 \
      nvidia-nvjitlink-cu12==12.8.93 \
      nvidia-cuda-nvrtc-cu12==12.8.93 \
      nvidia-cuda-cupti-cu12==12.8.90 \
      nvidia-nvtx-cu12==12.8.90 \
      nvidia-nccl-cu12 nvidia-nvshmem-cu12 nvidia-cufile-cu12 \
      nvidia-cusparselt-cu12 \
      torchvision==0.24.1 numpy pillow

# Every shim, built by scripts/build_client.sh. The maths libraries need
# replacing for the same reason the runtime does: each of them initialises
# through the driver's undocumented export tables.
COPY libcuda.so.1 libcudart.so.12 libcublas.so.12 libcublasLt.so.12 \
     libcudnn.so.9 /opt/rgpu/lib/
RUN cd /opt/rgpu/lib \
 && ln -s libcuda.so.1 libcuda.so \
 && ln -s libcudart.so.12 libcudart.so \
 && ln -s libcublas.so.12 libcublas.so \
 && ln -s libcublasLt.so.12 libcublasLt.so \
 && ln -s libcudnn.so.9 libcudnn.so

# LD_LIBRARY_PATH is not enough on its own: torch's libraries carry DT_RUNPATH,
# which the loader consults first, and torch preloads the CUDA libraries by
# absolute path besides. LD_PRELOAD wins over both.
ENV LD_LIBRARY_PATH=/opt/rgpu/lib
ENV LD_PRELOAD=/opt/rgpu/lib/libcudart.so.12:/opt/rgpu/lib/libcublas.so.12:/opt/rgpu/lib/libcublasLt.so.12:/opt/rgpu/lib/libcudnn.so.9:/opt/rgpu/lib/libcuda.so.1
ENV RGPU_SERVER=host.docker.internal:9713

WORKDIR /work
CMD ["python3", "-c", "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available())"]
