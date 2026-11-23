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

Buffer allocate(size_t elements, bool on_cpu, bool map_host = true) {
    Buffer buffer;
    const size_t bytes = elements * sizeof(Bf16);
    void* pointer = nullptr;
    if (on_cpu) {
        CHECK(cudaHostAlloc(&pointer, bytes, map_host ? cudaHostAllocMapped : cudaHostAllocDefault));
        buffer.cpu = static_cast<Bf16*>(pointer);
        if (!map_host) {
            return buffer;
        }
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

struct GemmShape {
    int rows;
    int n;
    int k;
};

struct PanelWorkspace {
    int panel_k;
    Bf16* b = nullptr;
    float* c = nullptr;
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

void gemm(cublasHandle_t handle, const Bf16* a, const Bf16* b, void* c, cudaDataType_t c_type,
          GemmShape shape, int a_stride, float beta) {
    const float alpha = 1.0f;
    // Row-major C = A B is column-major C^T = B^T A^T at the same addresses.
    CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, shape.n, shape.rows, shape.k, &alpha,
                       b, CUDA_R_16BF, shape.n, a, CUDA_R_16BF, a_stride, &beta,
                       c, c_type, shape.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

__global__ void cast_fp32_to_bf16(Bf16* out, const float* in, size_t elements) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elements;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        out[i] = __float2bfloat16_rn(in[i]);
    }
}

void run_gemm(cublasHandle_t handle, cudaStream_t stream, const Slot& slot, GemmShape shape,
              const PanelWorkspace& workspace) {
    if (workspace.panel_k == 0) {
        gemm(handle, slot.a.gpu, slot.b.gpu, slot.c.gpu, CUDA_R_16BF, shape, shape.k, 0.0f);
        return;
    }

    // Stream ordering protects the reusable B panel and FP32 accumulator.
    for (int start = 0; start < shape.k;) {
        const int span = std::min(workspace.panel_k, shape.k - start);
        const size_t b_offset = static_cast<size_t>(start) * shape.n;
        const Bf16* b_panel;
        if (workspace.b) {
            const size_t panel_bytes = static_cast<size_t>(span) * shape.n * sizeof(Bf16);
            CHECK(cudaMemcpyAsync(workspace.b, slot.b.cpu + b_offset, panel_bytes, cudaMemcpyHostToDevice, stream));
            b_panel = workspace.b;
        } else {
            b_panel = slot.b.gpu + b_offset;
        }
        gemm(handle, slot.a.gpu + start, b_panel, workspace.c, CUDA_R_32F,
             {shape.rows, shape.n, span}, shape.k, start == 0 ? 0.0f : 1.0f);
        start += span;
    }

    const size_t elements = static_cast<size_t>(shape.rows) * shape.n;
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((elements + threads - 1) / threads, 65535));
    cast_fp32_to_bf16<<<blocks, threads, 0, stream>>>(slot.c.gpu, workspace.c, elements);
    CHECK(cudaGetLastError());
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
    if (argc != 9 && argc != 10) {
        std::cerr << "Usage: cpu_gemm CASE M N K BATCH WARMUP ITERS SLOTS [K_PANEL]\n";
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
    const int panel_k = argc == 10 ? parse_int(argv[9], "K_PANEL") : 0;
    if (panel_k > k) {
        std::cerr << "K_PANEL must be <= K\n";
        return EXIT_FAILURE;
    }
    const int rows = m * batch;
    const GemmShape shape{rows, n, k};
    const size_t a_elements = static_cast<size_t>(rows) * k;
    const size_t b_elements = static_cast<size_t>(k) * n;
    const size_t c_elements = static_cast<size_t>(rows) * n;
    const bool stream_b = placement == 2 && panel_k > 0;
    const bool mapped_b = placement == 2 && !stream_b;

    // The launcher exposes the selected physical GPU as CUDA device 0.
    CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaDeviceProp device{};
    CHECK(cudaGetDeviceProperties(&device, 0));
    CHECK(cudaSetDevice(0));
    if ((mapped_b || placement == 3) && !device.canMapHostMemory) {
        std::cerr << "The selected GPU does not support mapped host memory\n";
        return EXIT_FAILURE;
    }
    std::cout << "gpu=" << device.name << " l2_bytes=" << device.l2CacheSize
              << " case=" << placement << " m=" << m << " n=" << n << " k=" << k << " b=" << batch
              << " slots=" << slots << " iterations=" << iterations;
    if (panel_k > 0) {
        std::cout << " k_panel=" << panel_k;
    }
    std::cout << std::endl;

    cudaStream_t stream;
    cublasHandle_t handle;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cublasCreate(&handle));
    CHECK(cublasSetStream(handle, stream));
    CHECK(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));

    PanelWorkspace workspace{panel_k};
    if (stream_b) {
        CHECK(cudaMalloc(&workspace.b, static_cast<size_t>(panel_k) * n * sizeof(Bf16)));
    }
    if (panel_k > 0) {
        CHECK(cudaMalloc(&workspace.c, c_elements * sizeof(float)));
    }
    std::vector<Slot> buffers(slots);
    for (int i = 0; i < slots; ++i) {
        Slot& slot = buffers[i];
        slot.a = allocate(a_elements, false);
        slot.b = allocate(b_elements, placement == 2, mapped_b);
        slot.c = allocate(c_elements, placement == 3);
        const uint64_t seed_a = 2ULL * i + 1;
        fill(slot.a, a_elements, seed_a, stream);
        fill(slot.b, b_elements, seed_a + 1, stream);
        if (slot.c.cpu) {
            std::memset(static_cast<void*>(slot.c.cpu), 0, c_elements * sizeof(Bf16));
        }
    }
    // A preflight launch initializes the library even when extra warmup is zero.
    run_gemm(handle, stream, buffers.front(), shape, workspace);
    for (int i = 0; i < warmup; ++i) {
        run_gemm(handle, stream, buffers[i % slots], shape, workspace);
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
        run_gemm(handle, stream, buffers[i % slots], shape, workspace);
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
    if (workspace.b) {
        CHECK(cudaFree(workspace.b));
    }
    if (workspace.c) {
        CHECK(cudaFree(workspace.c));
    }
    if (valid) {
        std::cout << "validation=passed\n";
    }
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
