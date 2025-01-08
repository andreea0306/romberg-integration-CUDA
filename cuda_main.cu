#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
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

/**
 * This kernel is the function f(x) = 1 / x
 * for the given element
 */
__global__ void f_function(const double* x, int* errorFlag, double* result, const int n, const int func_id) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        if (x[i] == 0) {
            *errorFlag = 1;
            //printf("NOK \n"); //debug
        } else {
            result[i] = evaluate_function(x[i], func_id);
        }
    }
}

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


/**
 * Function for safe compute -> to catch if it is devided to 0
 */
void safeCompute(double* host_x, double* host_result, int n, int func_id, cudaStream_t stream) {
    double* device_x = nullptr;
    double* device_result = nullptr;
    int* device_errorFlag = nullptr;

    int host_errorFlag = 0;

    cudaMalloc((void**)&device_x, n * sizeof(double));
    cudaMalloc((void**)&device_result, n * sizeof(double));
    cudaMalloc((void**)&device_errorFlag, sizeof(int));

    // Initialize errorFlag to 0
    cudaMemcpy(device_errorFlag, &host_errorFlag, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, host_x, n * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    f_function<<<blocksPerGrid, threadsPerBlock>>>(device_x, device_errorFlag, device_result, n, func_id);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete

    cudaMemcpy(host_result, device_result, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_errorFlag, device_errorFlag, sizeof(int), cudaMemcpyDeviceToHost);

    if (host_errorFlag != 0) {
        std::cerr << "[ERROR] Division by zero detected in kernel! \n";
    }

    cudaFree(device_x);
    cudaFree(device_result);
    cudaFree(device_errorFlag);
}
/*
double parallelSum(double* device_values, int n) {
    const int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t shared_mem_size = threadsPerBlock * sizeof(double);

    double* device_partial_sums = nullptr;
    cudaMalloc(&device_partial_sums, blocksPerGrid * sizeof(double));

    reduceSum<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(device_values, device_partial_sums, n);

    double* host_partial_sums = new double[blocksPerGrid];
    cudaMemcpy(host_partial_sums, device_partial_sums, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    double total_sum = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        total_sum += host_partial_sums[i];
    }

    delete[] host_partial_sums;
    cudaFree(device_partial_sums);
    return total_sum;
}
*/
double parallelSum(double* device_values, int n) {
    const int threadsPerBlock = 512;  // Increase threads per block if appropriate
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t shared_mem_size = threadsPerBlock * sizeof(double);

    double* device_partial_sums = nullptr;
    cudaMalloc(&device_partial_sums, blocksPerGrid * sizeof(double));

    // Perform reduction in parallel
    reduceSum<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(device_values, device_partial_sums, n);
    cudaDeviceSynchronize();  // Ensure kernel execution is complete

    // Use a single kernel to perform the final reduction step
    while (blocksPerGrid > 1) {
        int new_blocksPerGrid = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;
        reduceSum<<<new_blocksPerGrid, threadsPerBlock, shared_mem_size>>>(device_partial_sums, device_partial_sums, blocksPerGrid);
        blocksPerGrid = new_blocksPerGrid;
        cudaDeviceSynchronize();  // Ensure each reduction step is complete
    }

    // The final result is now in device_partial_sums[0]
    double result;
    cudaMemcpy(&result, device_partial_sums, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_partial_sums);
    return result;
}


void romberg_integration(double lower_limit, double upper_limit, uint32_t p, uint32_t q, int func_id, cudaStream_t stream) {
    
    if(lower_limit == 0 || upper_limit == 0) {
        std::cerr<<"[ERROR]: Division by zero detected! \n";
        return;
    }
    const int maxSize = 1001;
    double t[maxSize][maxSize] = {0};
    double h = upper_limit - lower_limit;
    

    t[0][0] = h / 2.0 * (1.0 / lower_limit + 1.0 / upper_limit);

    for (uint32_t i = 1; i <= p; i++) {
        uint32_t sl = static_cast<uint32_t>(pow(2, i - 1));
        double h_i = h / pow(2, i);
        double* x_points = new double[sl];
        double* f_values = new double[sl];

        for (uint32_t k = 1; k <= sl; k++) {
            x_points[k - 1] = lower_limit + (2 * k - 1) * h_i;
        }

        double* device_x_points = nullptr;
        double* device_f_values = nullptr;

        cudaMalloc(&device_x_points, sl * sizeof(double));
        cudaMalloc(&device_f_values, sl * sizeof(double));

        cudaMemcpy(device_x_points, x_points, sl * sizeof(double), cudaMemcpyHostToDevice);
        
        safeCompute(x_points, f_values, sl, func_id, stream);

        cudaMemcpy(device_f_values, f_values, sl * sizeof(double), cudaMemcpyHostToDevice);
        double sm = parallelSum(device_f_values, sl);

        delete[] x_points;
        delete[] f_values;
        cudaFree(device_x_points);
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
    uint32_t p = 10;
    uint32_t q = 10;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA-compatible GPU detected!" << std::endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "GPU Device " << i << ": " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    }
    int num_functions = 20;
    cudaStream_t streams[num_functions];
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<num_functions; i++) {
        cudaStreamCreate(&streams[i]);
        std::cout << "Integrating f" << i << "\n";
        romberg_integration(lower_limit, upper_limit, p, q, i+1, streams[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "\nExecution time: " << duration << " seconds\n";
    return 0;
}


