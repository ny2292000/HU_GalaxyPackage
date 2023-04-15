#include <stdio.h>
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include<vector>

void print(const std::vector<double> a) {
    std::cout << "The vector elements are : ";
    for (int i = 0; i < a.size(); i++)
        std::cout << a.at(i) << '\n';
}

__global__ void tensorial_product_kernel(double* v_i, double* v_j, int n_i, int n_j, double *v_ij) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_i && j < n_j) {
        int index = j  + i*n_j;
        v_ij[index] = v_i[i] * v_j[j];
        printf("The value of x is %f\n", v_ij[index]);
    }
}


double* createCUDAVector(const double* hostPointer, int n) {
    double* devicePointer;
    cudaMalloc((void**)&devicePointer, n * sizeof(double));
    cudaMemcpy(devicePointer, hostPointer, n * sizeof(double), cudaMemcpyHostToDevice);
    return devicePointer;
}


std::vector<double> zeros(int size) {
    return std::vector<double>(size, 0.0);
}


void deleteCUDAVector(double * x_vector_device){
    cudaFree(x_vector_device);
}

std::pair<dim3, dim3> get_block_size(int n_i, int n_j, int threads_per_block) {
    int n = n_i * n_j;
    int blocks_per_grid_x = (n_j + threads_per_block - 1) / threads_per_block;
    int blocks_per_grid_y = (n_i + threads_per_block - 1) / threads_per_block;
    dim3 block_size(threads_per_block, threads_per_block, 1);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y, 1);
    return std::make_pair(num_blocks, block_size);
}

int main() {
    double v_i[] = {1.0, 2.0, 3.0, 4.0};
    double v_j[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    int n_i = sizeof(v_i) / sizeof(double);
    int n_j = sizeof(v_j) / sizeof(double);

    double *v_i_device = createCUDAVector(v_i, n_i);
    double *v_j_device = createCUDAVector(v_j, n_j);
    double *dev_f_ij;
    double *v_ij;

    cudaMalloc((void**)&dev_f_ij, n_i * n_j * sizeof(double));
    v_ij = new double[n_i * n_j];


    int threads_per_block = 32;

// Get the block size for the tensor product kernel
    std::pair<dim3, dim3> block_info = get_block_size(n_i, n_j, threads_per_block);
    dim3 num_blocks = block_info.first;
    dim3 block_size = block_info.second;

//    dim3 block_size(32, 32, 1);
//    dim3 num_blocks((n_i + block_size.x - 1) / block_size.x,
//                    (n_j + block_size.y - 1) / block_size.y,
//                    1);

    tensorial_product_kernel<<<num_blocks, block_size>>>(v_i_device, v_j_device, n_i, n_j, dev_f_ij);

    // Copy result back to host
    cudaMemcpy(v_ij, dev_f_ij, n_i * n_j * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(v_i_device);
    cudaFree(v_j_device);
    cudaFree(dev_f_ij);

    // Print result
    for (int i = 0; i < n_i; i++) {
        for (int j = 0; j < n_j; j++) {
            printf("%f ", v_ij[i * n_j + j]);
        }
        printf("\n");
    }

    // Free host memory
    delete[] v_ij;

    return 0;
}
