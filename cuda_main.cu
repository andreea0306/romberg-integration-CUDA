#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <omp.h>
//#include "utils.h"

// cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);

__device__ double evaluate_function(double x, int func_id) {
    switch (func_id) {
        case 1: return 1/x;
        case 2: return x * x;
        case 3: return x * x * x;
        case 4: return pow(x, 4);
        case 5: return pow(x, 5);
        case 6: return x + 2 * x * x;
        case 7: return x - 3 * x * x + 2 * pow(x, 3);
        case 8: return 5 * x * x - 4 * x + 1;
        case 9: return x * x + 6 * x + 9;
        case 10: return x * x - 4 * x + 4;
        case 11: return 3 * x * x - 2 * x + 7;
        case 12: return pow(x, 3) - x * x + x - 1;
        case 13: return pow(x, 4) - 2 * pow(x, 3) + x * x - x + 3;
        case 14: return 2 * pow(x, 5) - 3 * pow(x, 3) + x - 5;
        case 15: return pow(x, 2) + 2 * x + 1;
        case 16: return 4 * pow(x, 4) - 2 * pow(x, 2) + 6 * x - 3;
        case 17: return x * x * x + 3 * x * x - x + 5;
        case 18: return 2 * pow(x, 6) - 4 * pow(x, 4) + x * x - 1;
        case 19: return pow(x, 7) - 3 * pow(x, 5) + 2 * pow(x, 3) - x + 4;
        case 20: return pow(x, 8) - 2 * pow(x, 6) + 3 * pow(x, 4) - 4 * x + 5;
        default: return 0.0;
    }
}

/*
 * CUDA kernel to compute function values for an array of inputs
 */
__global__ void f_function(const double* x, int* errorFlag, double* result, const int n, const int func_id) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        if (x[i] == 0) {
         	// Set error flag if input is zero (avoiding division by zero)
            *errorFlag = 1;
            //printf("NOK \n"); //debug
        } else {
            result[i] = evaluate_function(x[i], func_id);
        }
    }
}

/*
 * CUDA kernel to perform block-wise reduction (sum) of an array
 */
__global__ void reduceSum(const double* f_values, double* partial_sums, const int n) {
    extern __shared__ double shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    shared_mem[tid] = (idx < n) ? f_values[idx] : 0.0;
    __syncthreads();

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // Write the block's partial sum to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_mem[0];
    }
}

/*
 * Host function to safely launch the CUDA kernel and handle memory transfers
 */
void safeCompute(double* host_x, double* host_result, int n, int func_id, cudaStream_t stream) {
    double* device_x = nullptr;
    double* device_result = nullptr;
    int* device_errorFlag = nullptr;
    int host_errorFlag = 0;

	// Allocate device memory
    cudaMalloc((void**)&device_x, n * sizeof(double));
    cudaMalloc((void**)&device_result, n * sizeof(double));
    cudaMalloc((void**)&device_errorFlag, sizeof(int));

    // Initialize errorFlag to 0 and perform async memory transfers
    cudaMemcpyAsync(device_errorFlag, &host_errorFlag, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_x, host_x, n * sizeof(double), cudaMemcpyHostToDevice, stream);

    // Launch kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel in the given stream
    f_function<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(device_x, device_errorFlag, device_result, n, func_id);

    // Perform async memory transfers back to host (copy results back to host)
    cudaMemcpyAsync(host_result, device_result, n * sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&host_errorFlag, device_errorFlag, sizeof(int), cudaMemcpyDeviceToHost, stream);

    // Check for errors after stream synchronization
//    cudaStreamSynchronize(stream);
    if (host_errorFlag != 0) {
        std::cerr << "[ERROR] Division by zero detected in kernel! \n";
    }

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_result);
    cudaFree(device_errorFlag);
}

/*
 * Recursive parallel reduction to compute the sum of an array
 */
double parallelSum(double* device_values, int n, cudaStream_t stream) {
    const int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t shared_mem_size = threadsPerBlock * sizeof(double);

    double* device_partial_sums = nullptr;
    cudaMalloc(&device_partial_sums, blocksPerGrid * sizeof(double));

     // Perform reductions iteratively until one value remains
    reduceSum<<<blocksPerGrid, threadsPerBlock, shared_mem_size, stream>>>(device_values, device_partial_sums, n);
    while (blocksPerGrid > 1) {
      	printf("blocksPerGrid: %d\n", blocksPerGrid);
        int new_blocksPerGrid = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;
        reduceSum<<<new_blocksPerGrid, threadsPerBlock, shared_mem_size, stream>>>(device_partial_sums, device_partial_sums, blocksPerGrid);
        blocksPerGrid = new_blocksPerGrid;
    }

    double result = 0.0;
    cudaMemcpyAsync(&result, device_partial_sums, sizeof(double), cudaMemcpyDeviceToHost, stream);

    cudaFree(device_partial_sums);
//    cudaStreamSynchronize(stream); // Wait for all operations in the stream to complete
    return result;
}

/*
 * Host function to perform Romberg integration using CUDA streams
 */
void romberg_integration(double lower_limit, double upper_limit, uint32_t p, uint32_t q, int func_id, cudaStream_t stream) {
    if (lower_limit == 0 || upper_limit == 0) {
        std::cerr << "[ERROR]: Division by zero detected!\n";
        return;
    }

    const int maxSize = 1001;
    double t[maxSize][maxSize] = {0};
    double h = upper_limit - lower_limit;

    t[0][0] = h / 2.0 * (1.0 / lower_limit + 1.0 / upper_limit);

    for (uint32_t i = 1; i <= p; i++) {
        uint32_t sl = static_cast<uint32_t>(pow(2, i - 1));
        double h_i = h / pow(2, i);

        double* host_x_points = new double[sl];
        double* host_f_values = new double[sl];

        for (uint32_t k = 1; k <= sl; k++) {
            host_x_points[k - 1] = lower_limit + (2 * k - 1) * h_i;
        }

        safeCompute(host_x_points, host_f_values, sl, func_id, stream);

        double* device_f_values = nullptr;
        cudaMalloc(&device_f_values, sl * sizeof(double));
        cudaMemcpyAsync(device_f_values, host_f_values, sl * sizeof(double), cudaMemcpyHostToDevice, stream);

        double sm = parallelSum(device_f_values, sl, stream);

        delete[] host_x_points;
        delete[] host_f_values;
        cudaFree(device_f_values);

        t[i][0] = t[i - 1][0] / 2.0 + sm * h_i;
    }

    for (uint32_t c = 1; c <= p; c++) {
        for (uint32_t k = 1; k <= c && k <= q; k++) {
            uint32_t m = c - k;
            t[m + k][k] = (pow(4, k) * t[m + k][k - 1] - t[m + k - 1][k - 1]) / (pow(4, k) - 1);
        }
    }

    std::cout << "Romberg estimate of integration = " << t[p][q] << std::endl;
}


int main() {
    double lower_limit = -10.0;
    double upper_limit = 11.0;
    uint32_t p = 27;
    uint32_t q = 27;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    // Debug purpose 
    if (deviceCount == 0) {
        std::cout << "No CUDA-compatible GPU detected!" << std::endl;
        return 1;
    }

    // Debug purpose
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "GPU Device " << i << ": " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        if (prop.concurrentKernels) {
            std::cout << "Device supports concurrent kernel execution.\n";
        } else {
            std::cout << "Device does not support concurrent kernel execution.\n";
        }
    }

    const int num_functions = 20;
    cudaStream_t streams[num_functions];
    cudaEvent_t startEvents[num_functions], endEvents[num_functions];

    // Create CUDA streams and events
    for (int i = 0; i < num_functions; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&startEvents[i]);
        cudaEventCreate(&endEvents[i]);
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Launch integration tasks in parallel using streams
    for (int i = 0; i < num_functions; ++i) {
        // Record start of stream execution
        cudaEventRecord(startEvents[i], streams[i]);
        std::cout << "Integrating f" << (i + 1) << " in stream " << i << "\n";
        romberg_integration(lower_limit, upper_limit, p, q, i + 1, streams[i]);
        // Record end of stream execution
        cudaEventRecord(endEvents[i], streams[i]);
    }

    // Synchronize all streams to ensure tasks are complete
    for (int i = 0; i < num_functions; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    float f = 0.0;
    // Measure elapsed time for each stream and check overlap
    for (int i = 0; i < num_functions; ++i) {
        float elapsedTime = 0.0f;
        cudaEventElapsedTime(&elapsedTime, startEvents[i], endEvents[i]);
        std::cout << "Stream " << i << " execution time: " << elapsedTime << " ms\n";
        f += elapsedTime;
    }
    std::cout << "total time:" << f << std::endl;

    // Cleanup
    for (int i = 0; i < num_functions; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(startEvents[i]);
        cudaEventDestroy(endEvents[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "\nExecution time: " << duration << " seconds\n";

    return 0;
}

