// Compiled to a fatbin and loaded at run time by tests/cuda/vecadd.cpp.
//   nvcc -fatbin -o vecadd.fatbin vecadd_kernel.cu
//
// extern "C" keeps the name unmangled so cuModuleGetFunction can find it.
extern "C" __global__ void vecadd(const float* a, const float* b, float* c,
                                  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
