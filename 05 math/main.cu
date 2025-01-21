#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include "helper_cuda.h"
#include "CudaAllocator.h"
#include "ticktock.h"

template <class Func>
__global__ void parallel_for(int n, Func func) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x;
        i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    int n = 1<<28;
    std::vector<float, CudaAllocator<float>> gpu(n);
    std::vector<float> cpu(n);

    printf("%d\n", n);

    TICK(cpu_sinf);
    for(int i = 0; i < n; ++i) {
        cpu[i] = sinf(i);
    }
    TOCK(cpu_sinf);

    TICK(gpu_sinf);
    parallel_for<<<n / 512, 128>>>(n, [gpu = gpu.data()] __device__ (int i) {
        // sin()是double类型，计算float有性能损失
        // 1.0是double类型，1.0f是float类型
        // 还有更快的计算函数__sinf，但精确度降低
        // 编译器选项开启--use_fast_math降低精度换速度
        gpu[i] = sinf(i);
    });
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(gpu_sinf);
    return 0;
}