#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include "helper_cuda.h"

template <class T>
struct CudaAllocator{
    using value_type = T;

    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0) {
        checkCudaErrors(cudaFree(ptr));
    }

    // 可变参数模板
    template <class ...Args>
    // 这个构造是为了避免低效的零初始化
    void construct(T *p, Args &&...args) {
        // 编译期条件判断
        // 如果可变参数包大小为0且T是pod数据类型
        // 则跳过对象构造
        if constexpr(!(sizeof...(Args) == 0 && std::is_pod_v<T>)) {
            // placement new构造对象，参数列表采用完美转发保证左右值属性
            ::new((void *)p) T(std::forward<Args>(args)...);
        }
    }
};


// 核函数可以是一个模板函数
template <int N, class T>
__global__ void kernel(T *arr) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x;
        i < N; i += blockDim.x * gridDim.x) {
        arr[i] = i;
    }
}

template <class Func>
// 此处Func不接受指针，因为调用核函数指针在cpu栈上
__global__ void parallel_for(int n, Func func) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x;
        i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

// 核函数可以接受仿函数实现函数式编程
struct MyFunctor {
    __device__ void operator()(int i) const {
        printf("number %d\n", i);
    }
};

int main() {
    constexpr int n = 65536;
    // std::vector<T>隐藏第二模板参数，即std::vector<T, std::allocator<T>>
    // std::allocator<T>负责分配和释放内存，初始化T对象
    // std::allocator<T>有以下成员函数
    // T *allocate(size_t n) 继续调用malloc
    // void deallocate(T *p, size_t n) 继续调用free
    // 因此可以自己定义std::allocator<T>，管理分配和释放
    std::vector<int, CudaAllocator<int>> arr(n);

    kernel<n><<<32, 128>>>(arr.data());

    checkCudaErrors(cudaDeviceSynchronize());

    for(int i = 0; i < n; ++i) {
        printf("arr[%d]: %d\n", i, arr[i]);
    }

    // 核函数可以接受仿函数实现函数式编程
    parallel_for<<<32, 128>>>(n, MyFunctor{});
    checkCudaErrors(cudaDeviceSynchronize());

    // 仿函数可以是lambda表达式，但需要在cmake中添加flag
    // [&]捕获arr会报错，因为[&]捕获的是arr变量本身的指针，在CPU内存中
    // [=]捕获vector则是深拷贝
    // [arr = arr.data()]则是自定义捕获表达式，可以使用同一变量名
    // arr.data()指向vector第一个元素的地址
    parallel_for<<<32, 128>>>(n, [arr = arr.data()] __device__ (int i) {
        arr[i] = i;
    });
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}