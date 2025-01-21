#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include "helper_cuda.h"
#include "CudaAllocator.h"
#include "ticktock.h"

// atomicCAS可以实现任意CUDA没有提供的原子读-修改-写回指令
// 效率会低一些，有时候似乎会一直死循环？？？
__device__ __inline__ int my_atomic_add(int *dst, int src) {
    int old = *dst, expect;
    do {
        expect = old;
        old = atomicCAS(dst, expect, expect + src);
    } while(expect != old);
    return old;
}

__global__ void parallel_sum(int *sum, int const *arr, int n) {
    int local_sum = 0;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x;
        i < n; i += blockDim.x * gridDim.x) {
        // 由于加指令不是原子的，所以多线程寄存器写回时会存在覆盖问题
        // sum[0] += arr[i];
        // atomicXXX有一系列原子操作函数
        // atomicAdd(&sum[0], arr[i]);
        // 原子操作影响性能，所以通过TLS优化，即线程本地存储
        local_sum += arr[i];
    }
    my_atomic_add(&sum[0], local_sum);
}

__global__ void parallel_filter(int *sum, int *res, int const *arr, int n) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x;
        i < n; i += blockDim.x * gridDim.x) {
        if(arr[i] >= 2) {
            // atomicAdd会先将旧值赋值给loc，再Add
            int loc = atomicAdd(&sum[0], 1);
            res[loc] = arr[i];
        }
    }
}

int main() {
    int n = 65536, real_sum = 0;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(1);

    for(int i = 0; i < n; ++i) {
        arr[i] = std::rand() % 4;
    }

    TICK(cpu_sum)
    for(int i = 0; i < n; ++i) {
        real_sum += arr[i];
    }
    TOCK(cpu_sum);

    printf("real result: %d\n", real_sum);

    TICK(parallel_sum)
    parallel_sum<<<n / 4096, 128>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_sum);

    printf("result: %d\n", sum[0]);

    // 通过atomicAdd特性实现并行过滤器
    n = 1 << 24;
    std::vector<int, CudaAllocator<int>> arr_filter(n);
    std::vector<int, CudaAllocator<int>> sum_filter(1);
    std::vector<int, CudaAllocator<int>> res_filter(n);

    for(int i = 0; i < n; ++i) {
        arr_filter[i] = std::rand() % 4;
    }

    TICK(parallel_filter);
    parallel_filter<<< n / 4096, 512>>>(sum_filter.data(), res_filter.data(), arr_filter.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_filter);

    for(int i = 0; i < sum_filter[0]; ++i) {
        if(res_filter[i] < 2) {
            printf("Wrong At %d\n", i);
            return -1;
        }
    }
    printf("All correct!\n");
    return 0;
}