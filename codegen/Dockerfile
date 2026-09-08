# Codegen needs libclang and cuda.h, nothing else. The headers are extracted
# once by scripts/fetch_headers.sh and mounted in, so this image stays small
# instead of carrying the multi-gigabyte CUDA devel image.
FROM python:3.11-slim

# libc6-dev supplies stdlib.h, which cuda.h includes.
# clang supplies its own builtin headers (stddef.h and friends), which the
# pip libclang wheel does not ship.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libc6-dev clang \
 && rm -rf /var/lib/apt/lists/*

# The PyPI libclang wheel bundles the clang.cindex bindings and a matching
# native library, so no distro llvm packages are needed.
RUN pip install --no-cache-dir libclang==16.0.6

WORKDIR /src
ENTRYPOINT ["python3"]
