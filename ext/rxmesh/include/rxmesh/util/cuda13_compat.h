#pragma once
// Compatibility shims for CUDA 13+ (removed cub:: functors).
// Guarded so this is a no-op on CUDA 12.x.

#include <cub/cub.cuh>

#if CUDART_VERSION >= 13000

namespace cub {

struct Sum {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct Min {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

struct Max {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return (b > a) ? b : a;
    }
};

struct Inequality {
    template <typename T>
    __host__ __device__ __forceinline__ bool operator()(const T &a, const T &b) const {
        return a != b;
    }
};

struct Equality {
    template <typename T>
    __host__ __device__ __forceinline__ bool operator()(const T &a, const T &b) const {
        return a == b;
    }
};

}  // namespace cub

#endif  // CUDART_VERSION >= 13000
