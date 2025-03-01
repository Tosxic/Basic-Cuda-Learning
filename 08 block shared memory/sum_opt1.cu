#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include "helper_cuda.h"
#include "CudaAllocator.h"
#include "ticktock.h"


// 通过局部变量实现无原子操作方案
__global__ void parallel_sum(int *sum, int const *arr, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int local_sum = 0;
    for(int j = i * 1024; j < i * 1024 + 1024; ++j) {
        local_sum += arr[j];
    }
    sum[i] = local_sum;
}

int main() {
    int n = 1 << 24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(n / 1024);

    for(int i = 0; i < n; ++i) {
        arr[i] = i % 4;
    }

    TICK(parallel_sum);
    parallel_sum<<<n / 1024 / 128, 128>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());

    int final_sum = 0;
    for(int i = 0; i < n / 1024; ++i) {
        final_sum += sum[i];
    }
    TOCK(parallel_sum);

    printf("result: %d\n", final_sum);
    return 0;
}