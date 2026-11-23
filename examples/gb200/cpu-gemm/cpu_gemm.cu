// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using Bf16 = __nv_bfloat16;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << operation << " failed: " << cublasGetStatusString(status) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(call) check((call), #call)

int parse_int(std::string_view input, const char* name, int minimum = 1) {
    int value = 0;
    const char* end = input.data() + input.size();
    const auto result = std::from_chars(input.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end || value < minimum) {
        std::cerr << name << " must be an integer >= " << minimum << '\n';
        std::exit(EXIT_FAILURE);
    }
    return value;
}

struct Buffer {
    Bf16* gpu = nullptr;
    Bf16* cpu = nullptr;
};

Buffer allocate(size_t elements, bool on_cpu) {
    Buffer buffer;
    const size_t bytes = elements * sizeof(Bf16);
    void* pointer = nullptr;
    if (on_cpu) {
        CHECK(cudaHostAlloc(&pointer, bytes, cudaHostAllocMapped));
        buffer.cpu = static_cast<Bf16*>(pointer);
        CHECK(cudaHostGetDevicePointer(&pointer, buffer.cpu, 0));
    } else {
        CHECK(cudaMalloc(&pointer, bytes));
    }
    buffer.gpu = static_cast<Bf16*>(pointer);
    return buffer;
}

void release(Buffer buffer) {
    if (buffer.cpu) {
        CHECK(cudaFreeHost(buffer.cpu));
    } else {
        CHECK(cudaFree(buffer.gpu));
    }
}

struct Slot {
    Buffer a;
    Buffer b;
    Buffer c;
};

// Binary fractions keep the reference dot products exact at the default K.
__host__ __device__ int input_numerator(size_t index, uint64_t seed) {
    uint64_t bits = index + seed * 0x9e3779b97f4a7c15ULL;
    bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
    bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<int>((bits ^ (bits >> 31)) & 7) - 4;
}

__global__ void fill_device(Bf16* data, size_t elements, uint64_t seed) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        data[i] = __float2bfloat16_rn(input_numerator(i, seed) / 8.0f);
    }
}

void fill(Buffer buffer, size_t elements, uint64_t seed, cudaStream_t stream) {
    if (buffer.cpu) {
        for (size_t i = 0; i < elements; ++i) {
            buffer.cpu[i] = __float2bfloat16_rn(input_numerator(i, seed) / 8.0f);
        }
    } else {
        constexpr int threads = 256;
        const int blocks = static_cast<int>(std::min<size_t>((elements + threads - 1) / threads, 65535));
        fill_device<<<blocks, threads, 0, stream>>>(buffer.gpu, elements, seed);
        CHECK(cudaGetLastError());
    }
}

void gemm(cublasHandle_t handle, const Slot& slot, int rows, int n, int k) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C = A B is column-major C^T = B^T A^T at the same addresses.
    CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, rows, k, &alpha,
                       slot.b.gpu, CUDA_R_16BF, n, slot.a.gpu, CUDA_R_16BF, k, &beta,
                       slot.c.gpu, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

struct Sample {
    int row;
    int col;
};

std::array<Sample, 12> sample_points(int rows, int n) {
    // Include boundary rows and columns to exercise tail tiles for custom sizes.
    std::array<Sample, 12> points;
    size_t index = 0;
    for (int row : {0, rows / 3, rows / 2, rows - 1}) {
        for (int col : {0, n / 2, n - 1}) {
            points[index++] = {row, col};
        }
    }
    return points;
}

bool validate(const Slot& slot, int slot_index, const std::array<Sample, 12>& points, int n, int k) {
    const uint64_t seed_a = 2ULL * slot_index + 1;
    const uint64_t seed_b = seed_a + 1;
    for (const auto& [row, col] : points) {
        int64_t sum = 0;
        for (int inner = 0; inner < k; ++inner) {
            sum += input_numerator(static_cast<size_t>(row) * k + inner, seed_a) *
                   input_numerator(static_cast<size_t>(inner) * n + col, seed_b);
        }
        const float expected = __bfloat162float(__float2bfloat16_rn(static_cast<float>(sum) / 64.0f));
        const size_t index = static_cast<size_t>(row) * n + col;
        Bf16 result;
        if (slot.c.cpu) {
            result = slot.c.cpu[index];
        } else {
            CHECK(cudaMemcpy(&result, slot.c.gpu + index, sizeof(result), cudaMemcpyDeviceToHost));
        }
        const float actual = __bfloat162float(result);
        // With this input every FP32 partial sum is exact up to this bound on K.
        const float tolerance = k <= (1 << 20) ? 0.0f : std::abs(expected) / 128.0f + 1.0f / 64.0f;
        if (!(std::abs(actual - expected) <= tolerance)) {
            std::cerr << "Validation failed for slot " << slot_index << ", C[" << row << "," << col
                      << "] = " << actual << ", expected " << expected << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "Usage: cpu_gemm CASE M N K BATCH WARMUP ITERS SLOTS\n";
        return EXIT_FAILURE;
    }
    const int placement = parse_int(argv[1], "CASE");
    const int m = parse_int(argv[2], "M");
    const int n = parse_int(argv[3], "N");
    const int k = parse_int(argv[4], "K");
    const int batch = parse_int(argv[5], "BATCH");
    const int warmup = parse_int(argv[6], "WARMUP", 0);
    const int iterations = parse_int(argv[7], "ITERS");
    const int slots = parse_int(argv[8], "SLOTS");
    if (placement > 3 || m > std::numeric_limits<int>::max() / batch) {
        std::cerr << "CASE must be 1, 2 or 3 and M * BATCH must fit in an int\n";
        return EXIT_FAILURE;
    }
    const int rows = m * batch;
    const size_t a_elements = static_cast<size_t>(rows) * k;
    const size_t b_elements = static_cast<size_t>(k) * n;
    const size_t c_elements = static_cast<size_t>(rows) * n;

    // The launcher exposes the selected physical GPU as CUDA device 0.
    CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaDeviceProp device{};
    CHECK(cudaGetDeviceProperties(&device, 0));
    CHECK(cudaSetDevice(0));
    if (placement != 1 && !device.canMapHostMemory) {
        std::cerr << "The selected GPU does not support mapped host memory\n";
        return EXIT_FAILURE;
    }
    std::cout << "gpu=" << device.name << " l2_bytes=" << device.l2CacheSize
              << " case=" << placement << " m=" << m << " n=" << n << " k=" << k << " b=" << batch
              << " slots=" << slots << " iterations=" << iterations << std::endl;

    cudaStream_t stream;
    cublasHandle_t handle;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cublasCreate(&handle));
    CHECK(cublasSetStream(handle, stream));
    CHECK(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));

    std::vector<Slot> buffers(slots);
    for (int i = 0; i < slots; ++i) {
        Slot& slot = buffers[i];
        slot.a = allocate(a_elements, false);
        slot.b = allocate(b_elements, placement == 2);
        slot.c = allocate(c_elements, placement == 3);
        const uint64_t seed_a = 2ULL * i + 1;
        fill(slot.a, a_elements, seed_a, stream);
        fill(slot.b, b_elements, seed_a + 1, stream);
        if (slot.c.cpu) {
            std::memset(static_cast<void*>(slot.c.cpu), 0, c_elements * sizeof(Bf16));
        }
    }
    // A preflight launch initializes the library even when extra warmup is zero.
    gemm(handle, buffers.front(), rows, n, k);
    for (int i = 0; i < warmup; ++i) {
        gemm(handle, buffers[i % slots], rows, n, k);
    }
    CHECK(cudaStreamSynchronize(stream));

    const auto points = sample_points(rows, n);
    const int measured_slots = std::min(iterations, slots);
    // Require the measured launches to overwrite the sampled output after warmup.
    for (int i = 0; i < measured_slots; ++i) {
        Buffer output = buffers[i].c;
        for (const auto& [row, col] : points) {
            const size_t index = static_cast<size_t>(row) * n + col;
            if (output.cpu) {
                output.cpu[index] = __float2bfloat16_rn(std::numeric_limits<float>::quiet_NaN());
            } else {
                CHECK(cudaMemsetAsync(output.gpu + index, 0xff, sizeof(Bf16), stream));
            }
        }
    }
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaProfilerStart());
    for (int i = 0; i < iterations; ++i) {
        gemm(handle, buffers[i % slots], rows, n, k);
    }
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaProfilerStop());

    bool valid = true;
    for (int i = 0; i < measured_slots && valid; ++i) {
        valid = validate(buffers[i], i, points, n, k);
    }
    CHECK(cublasDestroy(handle));
    CHECK(cudaStreamDestroy(stream));
    for (const Slot& slot : buffers) {
        release(slot.a);
        release(slot.b);
        release(slot.c);
    }
    if (valid) {
        std::cout << "validation=passed\n";
    }
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
