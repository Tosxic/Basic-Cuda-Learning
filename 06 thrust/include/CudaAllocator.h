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