#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include "helper_cuda.h"
#include "CudaAllocator.h"
#include "ticktock.h"
#include <thrust/universal_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/for_each.h>

template <class Func>
__global__ void parallel_for(int n, Func func) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x;
        i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    int n = 65536;
    float a = 3.14f;
    // 分配到统一内存，GPU和CPU都能访问
    thrust::universal_vector<float> x(n);
    thrust::universal_vector<float> y(n);
    
    for(int i = 0; i < n; ++i) {
        x[i] = std::rand() * (1.f / RAND_MAX);
        y[i] = std::rand() * (1.f / RAND_MAX);
    }

    parallel_for<<<n / 512, 128>>>(n, [a, x = x.data(), y = y.data()] __device__ (int i) {
        x[i] = a * x[i] + y[i];
    });

    // 分配到CPU内存
    thrust::host_vector<float> x_host(n);
    thrust::host_vector<float> y_host(n);
    // 分配到GPU内存
    thrust::device_vector<float> x_dev(n);
    thrust::device_vector<float> y_dev(n);

    // thrust::generate生成一系列数
    auto float_rand = [] {
        return std::rand() * (1.f / RAND_MAX);
    };
    thrust::generate(x_host.begin(), x_host.end(), float_rand);
    thrust::generate(y_host.begin(), y_host.end(), float_rand);

    parallel_for<<<n / 512, 128>>>(n, [a, x_dev = x.data(), y_dev = y.data()] __device__ (int i) {
        x_dev[i] = a * x_dev[i] + y_dev[i];
    });

    // thrust::for_each逐元素访问
    thrust::for_each(x_dev.begin(), x_dev.end(), [] __device__ (float &x) {
        x += 100.0f;
    });
    // 相当于进行内存拷贝，有同步功能
    x_host = x_dev;
    for(int i = 0; i < n; ++i) {
        printf("x[%d] = %f\n", i, x[i]);
    }
    // 通过for_each构建计数迭代器
    thrust::for_each(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(10),
        [] __device__ (int i) {
            printf("%d ", i);
    });
    printf("\n");
    // 迭代器合并
    thrust::for_each(
        thrust::make_zip_iterator(x_dev.begin(), y_dev.cbegin()),
        thrust::make_zip_iterator(x_dev.end(), y_dev.cend()),
        [a] __device__ (auto const &tup) {
        auto &x = thrust::get<0>(tup);
        auto const &y = thrust::get<1>(tup);
        x = a * x + y;
    });
    return 0;
}