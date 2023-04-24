//============================================================================
// Name        : GalaxyFormation.cpp
// Author      : Marco Pereira
// Version     : 1.0.0
// Copyright   : Your copyright notice
// Description : Hypergeometrical Universe Galaxy Formation in C++, Ansi-style
//============================================================================
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include <vector>
#include <memory>  // for std::unique_ptr
#include <cmath>
#include <stdio.h>
#include <stdexcept>
#include <cstring>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include <array>
#include "/usr/include/boost/python.hpp"
#include <iostream>
#include <future>
#include <nlopt.hpp>
#include <string>
#define MAX_GRID_DATA_SIZE 800


__constant__ float const_grid_data[MAX_GRID_DATA_SIZE];

float pi = 3.141592653589793;



void print(const std::vector<float> a) {
    std::cout << "The vector elements are : ";
    for (int i = 0; i < a.size(); i++)
        std::cout << std::scientific << a.at(i) << '\n';
}


void deleteCUDAVector(float *x_vector_host, float *x_vector_device) {
    delete[] x_vector_host;
    cudaFree(x_vector_device);
}

std::pair<dim3, dim3> get_block_size(int n, int threads_per_block) {
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    dim3 block_size(threads_per_block, 1, 1);
    dim3 num_blocks(blocks_per_grid, 1, 1);
    return std::make_pair(num_blocks, block_size);
}


float massCalcX(float alpha, float rho, float h, float x) {
    float factor = 0.0007126927557971729; // factor takes care of moving from rho as atom/cc to kg/lyr^3, with alpha = 1/lyr and h0 = in lyr div sun_mass
    float M_si = -2 * pi * h * rho * x * exp(-alpha * x) / alpha - 2 * pi * h * rho * exp(-alpha * x) / pow(alpha, 2) +
                  2 * pi * h * rho / pow(alpha, 2);
    M_si = M_si * factor;
    return M_si;
}


float massCalc(float alpha, float rho, float h) {
    float factor = 0.0007126927557971729; // factor takes care of moving from rho as atom/cc to kg/lyr^3, with alpha = 1/lyr and h0 = in lyr div sun_mass
    float Mtotal_si = 2 * pi * h * rho / pow(alpha, 2);
    Mtotal_si = Mtotal_si * factor;
    return Mtotal_si;
}


std::vector<float> vec_from_array(PyArrayObject *array) {
    // Check that input is a 1-dimensional array of floats
    if (PyArray_NDIM(array) != 1 || PyArray_TYPE(array) != NPY_FLOAT) {
        throw std::invalid_argument("Input must be a 1D NumPy array of floats");
    }

    // Get the size of the array and a pointer to its data
    int size = PyArray_SIZE(array);
    float *data_ptr = static_cast<float *>(PyArray_DATA(array));

    // Create a vector from the array data
    std::vector<float> vec(data_ptr, data_ptr + size);

    return vec;
}


PyArrayObject *array_from_vec(std::vector<float> vec) {
    // Create a 1D NumPy array of the same size as the input vector
    npy_intp size = vec.size();
    PyObject * array = PyArray_SimpleNew(1, &size, NPY_FLOAT);

    // Copy the input vector data to the array data
    float *data_ptr = static_cast<float *>(PyArray_DATA(reinterpret_cast<PyArrayObject *>(array)));
    memcpy(data_ptr, vec.data(), size * sizeof(float));

    return reinterpret_cast<PyArrayObject *>(array);
}

std::vector<float> costhetaFunc(const std::vector<float> &theta) {
    unsigned int points = theta.size();
    std::vector<float> res(points);
    for (unsigned int i = 0; i < points; i++) {
        res[i] = std::cos(theta[i]);
    }
    return res;
}

std::vector<float> sinthetaFunc(const std::vector<float> &theta) {
    unsigned int points = theta.size();
    std::vector<float> res(points);
    for (unsigned int i = 0; i < points; i++) {
        res[i] = sin(theta[i]);
    }
    return res;
}


std::vector<float> linspace(float start, float end, size_t points) {
    std::vector<float> res(points);
    float step = (end - start) / (points - 1);
    size_t i = 0;
    for (auto &e: res) {
        e = start + step * i++;
    }
    return res;
}

std::vector<float> density(float rho_0, float alpha_0, float rho_1, float alpha_1, std::vector<float> r) {
    unsigned int vecsize = r.size();
    std::vector<float> density_(vecsize);
    // to kg/lyr^3
    rho_0 *= 1.4171253E27;  //(h_mass/uu.cm**3).to(uu.kg/uu.lyr**3) =<Quantity 1.41712531e+27 kg / lyr3>
    rho_1 *= 1.4171253E27;
    for (unsigned int i = 0; i < vecsize; i++) {
        density_[i] = rho_0 * exp(-alpha_0 * r[i]) + rho_1 * exp(-alpha_1 * r[i]);
    }
    return density_;
}


// Returns a vector of zeros with the given size
std::vector<float> zeros(int size) {
    return std::vector<float>(size, 0.0);
}


float *createCUDAVector(const float *hostPointer, int n) {
    float *devicePointer;
    cudaMalloc((void **) &devicePointer, n * sizeof(float));
    cudaMemcpy(devicePointer, hostPointer, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Error copying host pointer to device: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    return devicePointer;
}

float *createCUDAVector(const std::vector<float> &hostVector) {
    float *devicePointer;
    cudaMalloc((void **) &devicePointer, hostVector.size() * sizeof(float));
    cudaMemcpy(devicePointer, hostVector.data(), hostVector.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "Error copying host vector to device: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
    return devicePointer;
}


void deleteCUDAVector(float *x_vector_device) {
    cudaFree(x_vector_device);
}

std::pair<dim3, dim3> get_block_size(int n_i, int n_j, int threads_per_block) {
    int blocks_per_grid_x = (n_j + threads_per_block - 1) / threads_per_block;
    int blocks_per_grid_y = (n_i + threads_per_block - 1) / threads_per_block;
    dim3 block_size(threads_per_block, threads_per_block, 1);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y, 1);
    return std::make_pair(num_blocks, block_size);
}

//std::pair<dim3, dim3> get_block_size3(int n_i, int n_j, int n_k, int threads_per_block) {
//    int blocks_per_grid_x = (n_j + threads_per_block - 1) / threads_per_block;
//    int blocks_per_grid_y = (n_i + threads_per_block - 1) / threads_per_block;
//    int blocks_per_grid_z = (n_k + threads_per_block - 1) / threads_per_block;
//    dim3 block_size(threads_per_block, threads_per_block, threads_per_block);
//    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y, threads_per_block);
//    return std::make_pair(num_blocks, block_size);
//}


void check_device_memory_allocation(const void *device_pointer, const std::string &name) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error: Unable to allocate memory on device for " << name << std::endl;
        std::exit(EXIT_FAILURE);
    } else {
        std::cout << "Allocated memory on device for " << name << std::endl;
    }
}

__device__ double atomicAddDouble(float* address, float val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ float atomicAddFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val + __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}






// CUDA kernel to compute the gravitational acceleration f_z
// for points in r_sampling and z_sampling
__global__ void get_all_g_kernel(int nr, int nz, int ntheta, int nr_sampling, int nz_sampling, float G,
                                 bool radial, float *f_z, bool debug) {

    // Get the indices of the point in r and z for this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // r
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // z
    if (i < nr_sampling && j < nz_sampling) {
        int idx = i + j * nr_sampling;
        // Initialize the result variable for this thread
        float res = 0.0f;
        float thisres = 0.0f;
        // Loop over all r, z, and theta points in the grid
        for (int ir = 0; ir < nr; ir++) {
            float r_ir = const_grid_data[i];
            float dv0_ir = const_grid_data[nr + nz + 2 * ntheta + i];
            float rho_ir = const_grid_data[2 * nr + nz + 2 * ntheta + i];
            if (radial && (r_ir > const_grid_data[3 * nr + nz + 2 * ntheta + i])) {
                break;
            }
            printf("GPU");
            for (int iz = 0; iz < nz; iz++) {
                float z_iz = const_grid_data[nr + j];
                for (int k = 0; k < ntheta; k++) {
                    float costheta_k = const_grid_data[nr + nz + k];
                    float sintheta_k = const_grid_data[nr + nz + ntheta + k];
                    float r_sampling_i = const_grid_data[3 * nr + nz + 2 * ntheta + i];
                    float z_sampling_j = const_grid_data[3 * nr + nr_sampling + nz + 2 * ntheta + j];

                    // Compute the distance between the sampling point and the point in r and z for this thread
                    float d_2 = (z_iz - z_sampling_j) * (z_iz - z_sampling_j) +
                                 (r_sampling_i - r_ir * sintheta_k) * (r_sampling_i - r_ir * sintheta_k) +
                                 r_ir * r_ir * costheta_k * costheta_k;
                    float d_1 = sqrt(d_2);
                    float d_3 = d_1 * d_1 * d_1;

                    //                    d = pow(z[j] - z_sampling_jj, 2.0) + pow(r_sampling_ii - r[i] * sintheta[k], 2.0) +
//                        r[i] * r[i] * costheta[k] * costheta[k];
                    if (radial) {
                        thisres = G * rho_ir * r_ir * dv0_ir * (r_sampling_i - r_ir * sintheta_k) / d_3;
//                        thisres = G * rho[i] * r[i] * dv0[i] * (r_sampling_ii - r[i] * sintheta[k]) / pow(d, 1.5);
                        res += thisres;
                    } else {
                        thisres = G * rho_ir * r_ir * dv0_ir * (z_iz - z_sampling_j) / d_3;;
                        res += thisres;
//                        thisres = G * rho[i] * r[i] * dv0[i] * (z[j] - z_sampling_jj) / pow(d, 1.5);
                    }

                    if (debug) {
                        if (ir == 5 && iz == 5 && k == 5) {
//                            std::vector<float> shared_data(r);
//                            shared_data.insert(shared_data.end(), z.begin(), z.end());
//                            shared_data.insert(shared_data.end(), costheta.begin(), costheta.end());
//                            shared_data.insert(shared_data.end(), sintheta.begin(), sintheta.end());
//                            shared_data.insert(shared_data.end(), dv0.begin(), dv0.end());
//                            shared_data.insert(shared_data.end(), rho.begin(), rho.end());
//                            shared_data.insert(shared_data.end(), r_sampling.begin(), r_sampling.end());
//                            shared_data.insert(shared_data.end(), z_sampling.begin(), z_sampling.end());
                            printf("GPU \n");
                            printf("The value of f_z is %e\n", thisres);
                            printf("The value of distance is %fd\n", d_1);
                            printf("The value of r[i] is %fd\n", r_ir);
                            printf("The value of z[j] is %fd\n", z_iz);
                            printf("The value of costheta is %fd\n", costheta_k);
                            printf("The value of sintheta is %fd\n", sintheta_k);
                            printf("The value of dv0 is %fd\n", dv0_ir);
                            printf("The value of rho is %e\n", rho_ir);
                            printf("The value of rsampling is %fd\n", r_sampling_i);
                            printf("The value of zsampling is %fd\n", z_sampling_j);
                            printf("The value of G is %e\n", G);
                        }
                    }
                }
            }
        }
        atomicAdd(&f_z[idx], res);
    }
}

std::vector<float> get_all_g_impl_cuda(float G, const std::vector<float> &dv0, const std::vector<float> &r_sampling,
                                        const std::vector<float> &z_sampling, const std::vector<float> &r,
                                        const std::vector<float> &z,
                                        const std::vector<float> &costheta, const std::vector<float> &sintheta,
                                        const std::vector<float> &rho, bool debug,
                                        bool radial = true) {
    // Combine r, z, costheta, sintheta, and dv0 into a single vector (grid_data) for cudaMemcpy
    std::vector<float> grid_data(r);
    grid_data.insert(grid_data.end(), z.begin(), z.end());
    grid_data.insert(grid_data.end(), costheta.begin(), costheta.end());
    grid_data.insert(grid_data.end(), sintheta.begin(), sintheta.end());
    grid_data.insert(grid_data.end(), dv0.begin(), dv0.end());
    grid_data.insert(grid_data.end(), rho.begin(), rho.end());
    grid_data.insert(grid_data.end(), r_sampling.begin(), r_sampling.end());
    grid_data.insert(grid_data.end(), z_sampling.begin(), z_sampling.end());

    if (grid_data.size() > MAX_GRID_DATA_SIZE) {
        // Handle error, e.g., return an empty vector or throw an exception
        std::cout << "too large grid_data ";
        return std::vector<float>();
    }

    int nr_sampling = r_sampling.size();
    int nz_sampling = z_sampling.size();
    int nr = r.size();
    int nz = z.size();

    // Copy grid_data to const_grid_data
    cudaError_t err = cudaMemcpyToSymbol(const_grid_data, grid_data.data(), grid_data.size() * sizeof(float));
    if (err != cudaSuccess) {
        printf("Error copying data to constant memory: %s\n", cudaGetErrorString(err));
        return std::vector<float>();
    }



    // Allocate and copy device memory
    int n_sampling = nr_sampling * nz_sampling; // Change this to the desired size
    float* f_z = new float[n_sampling];
    std::fill_n(f_z, n_sampling, 0.0);
    float *dev_f_z = createCUDAVector(f_z, n_sampling);


// Get the block size for the tensor product kernel
    // Get the block size for the tensor product kernel
    int ntheta = costheta.size();
    dim3 block_size(16, 16, 1);
    dim3 grid_size((nr + block_size.x - 1) / block_size.x,
                   (nz + block_size.y - 1) / block_size.y,
                   (ntheta + block_size.z - 1) / block_size.z);


    cudaDeviceProp prop;
    int device_id=0;
    cudaGetDeviceProperties(&prop, device_id);
    int max_threads_per_block = prop.maxThreadsPerBlock;



    get_all_g_kernel<<<grid_size, block_size>>>(nr, nz, ntheta, nr_sampling, nz_sampling, G,
                                                radial, dev_f_z, debug);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return std::vector<float>();
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Device synchronization error: %s\n", cudaGetErrorString(err));
        return std::vector<float>();
    }

    // Copy results back to host
    std::vector<float> f_z_vec(n_sampling);
    err = cudaMemcpy(f_z_vec.data(), dev_f_z, n_sampling * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Device synchronization error: %s\n", cudaGetErrorString(err));
        return std::vector<float>();
    }

    // Free device memory
    cudaFree(dev_f_z);
    delete[] f_z;

    // Return result
    return f_z_vec;
}


//
//__global__ void get_all_g_kernelOld(int nr, int nz, int ntheta, int nr_sampling, int nz_sampling, float G,
//                                 bool radial, float *f_z, bool debug) {
//
//    // Get the indices of the point in r and z for this thread
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (i < nr_sampling && j < nz_sampling) {
//        int idx = i + j * nr_sampling;
//        // Initialize the result variable for this thread
//        float res = 0.0;
//        float thisres = 0.0;
//        // Loop over all r and z points in the grid
//        for (int ir = 0; ir < nr; ir++) {
//            float r_ir = const_grid_data[ir];
//            float dv0_ir = const_grid_data[nr + nz + 2 * ntheta + ir];
//            float rho_ir = const_grid_data[2 * nr + nz + 2 * ntheta + ir];
//            if (radial && (r_ir > const_grid_data[3 * nr + nz + 2 * ntheta + i])) {
//                break;
//            }
//            for (int iz = 0; iz < nz; iz++) {
//                float z_iz = const_grid_data[nr + iz];
//                for (int k = 0; k < ntheta; k++) {
//                    float costheta_k = const_grid_data[nr + nz + k];
//                    float sintheta_k = const_grid_data[nr + nz + ntheta + k];
//                    float r_sampling_i = const_grid_data[3 * nr + nz + 2 * ntheta + i];
//                    float z_sampling_j = const_grid_data[3 * nr + nr_sampling + nz + 2 * ntheta + j];
//
//                    // Compute the distance between the sampling point and the point in r and z for this thread
//                    float d_2 = (z_iz - z_sampling_j) * (z_iz - z_sampling_j) +
//                                 (r_sampling_i - r_ir * sintheta_k) * (r_sampling_i - r_ir * sintheta_k) +
//                                 r_ir * r_ir * costheta_k * costheta_k;
//                    float d_1 = sqrt(d_2);
//                    float d_3 = d_1 * d_1 * d_1;
//
//                    //                    d = pow(z[j] - z_sampling_jj, 2.0) + pow(r_sampling_ii - r[i] * sintheta[k], 2.0) +
////                        r[i] * r[i] * costheta[k] * costheta[k];
//                    if (radial) {
//                        thisres= G * rho_ir * r_ir * dv0_ir *(r_sampling_i - r_ir * sintheta_k)/d_3;
////                        thisres = G * rho[i] * r[i] * dv0[i] * (r_sampling_ii - r[i] * sintheta[k]) / pow(d, 1.5);
//                        res += thisres;
//                    } else {
//                        thisres = G * rho_ir * r_ir * dv0_ir *(z_iz - z_sampling_j )/d_3;;
//                        res += thisres;
////                        thisres = G * rho[i] * r[i] * dv0[i] * (z[j] - z_sampling_jj) / pow(d, 1.5);
//                    }
//
//                    if(debug){
//                        if (ir==5 && iz==5 && k == 5) {
////                            std::vector<float> shared_data(r);
////                            shared_data.insert(shared_data.end(), z.begin(), z.end());
////                            shared_data.insert(shared_data.end(), costheta.begin(), costheta.end());
////                            shared_data.insert(shared_data.end(), sintheta.begin(), sintheta.end());
////                            shared_data.insert(shared_data.end(), dv0.begin(), dv0.end());
////                            shared_data.insert(shared_data.end(), rho.begin(), rho.end());
////                            shared_data.insert(shared_data.end(), r_sampling.begin(), r_sampling.end());
////                            shared_data.insert(shared_data.end(), z_sampling.begin(), z_sampling.end());
//                            printf("GPU \n");
//                            printf("The value of f_z is %e\n", thisres);
//                            printf("The value of distance is %fd\n", d_1);
//                            printf("The value of r[i] is %fd\n", r_ir);
//                            printf("The value of z[j] is %fd\n", z_iz);
//                            printf("The value of costheta is %fd\n", costheta_k);
//                            printf("The value of sintheta is %fd\n", sintheta_k);
//                            printf("The value of dv0 is %fd\n", dv0_ir);
//                            printf("The value of rho is %e\n", rho_ir);
//                            printf("The value of rsampling is %fd\n", r_sampling_i);
//                            printf("The value of zsampling is %fd\n", z_sampling_j);
//                            printf("The value of G is %e\n", G);
//                        }
//                    }
//
//                }
//            }
//        }
//        // Store the result variable for this thread in the output array
//        f_z[idx] = res;
//    }
//}
//
//
//std::vector<float> get_all_g_impl_cudaOld(float G, const std::vector<float> &dv0, const std::vector<float> &r_sampling,
//                                        const std::vector<float> &z_sampling, const std::vector<float> &r,
//                                        const std::vector<float> &z,
//                                        const std::vector<float> &costheta, const std::vector<float> &sintheta,
//                                        const std::vector<float> &rho, bool debug,
//                                        bool radial = true) {
//    // Combine r, z, costheta, sintheta, and dv0 into a single vector (grid_data) for cudaMemcpy
//    std::vector<float> grid_data(r);
//    grid_data.insert(grid_data.end(), z.begin(), z.end());
//    grid_data.insert(grid_data.end(), costheta.begin(), costheta.end());
//    grid_data.insert(grid_data.end(), sintheta.begin(), sintheta.end());
//    grid_data.insert(grid_data.end(), dv0.begin(), dv0.end());
//    grid_data.insert(grid_data.end(), rho.begin(), rho.end());
//    grid_data.insert(grid_data.end(), r_sampling.begin(), r_sampling.end());
//    grid_data.insert(grid_data.end(), z_sampling.begin(), z_sampling.end());
//
//    if (grid_data.size() > MAX_GRID_DATA_SIZE) {
//        // Handle error, e.g., return an empty vector or throw an exception
//        std::cout << "too large grid_data ";
//        return std::vector<float>();
//    }
//
//    int nr_sampling = r_sampling.size();
//    int nz_sampling = z_sampling.size();
//    int nr = r.size();
//    int nz = z.size();
//
//    // Copy grid_data to const_grid_data
//    cudaError_t err = cudaMemcpyToSymbol(const_grid_data, grid_data.data(), grid_data.size() * sizeof(float));
//    if (err != cudaSuccess) {
//        printf("Error copying data to constant memory: %s\n", cudaGetErrorString(err));
//        return std::vector<float>();
//    }
//
//
//
//    // Allocate and copy device memory
//    float *f_z = new float(nr_sampling * nz_sampling);
////    float *dev_grid_data = createCUDAVector(grid_data);
////    float *dev_rho = createCUDAVector(rho);
//    float *dev_f_z = createCUDAVector(f_z, nr_sampling * nz_sampling);
////    float *dev_dv0 = createCUDAVector(dv0);
//
//
//
//// Get the block size for the tensor product kernel
//    // Get the block size for the tensor product kernel
//    int ntheta = costheta.size();
//    int threads_per_block = 32;
//    std::pair<dim3, dim3> block_info = get_block_size(nr_sampling * nz_sampling, threads_per_block);
//    dim3 num_blocks = block_info.first;
//    dim3 block_size = block_info.second;
//
//
//    // Launch kernel
//    get_all_g_kernel<<<num_blocks, block_size>>>(nr, nz,ntheta, nr_sampling, nz_sampling, G,
//                                                 radial, dev_f_z, debug);
//    // Copy results back to host
//    std::vector<float> f_z_vec(nr_sampling * nz_sampling);
//    cudaMemcpy(f_z_vec.data(), dev_f_z, nr_sampling * nz_sampling * sizeof(float), cudaMemcpyDeviceToHost);
//
//
//    // Free device memory
//    cudaFree(dev_f_z);
//
//    // Return result
//    return f_z_vec;
//}

// # CPU functions

float
get_g_cpu(float r_sampling_ii, float z_sampling_jj, float G, const std::vector<float> &dv0,
          const std::vector<float> &r,
          const std::vector<float> &z,
          const std::vector<float> &costheta, const std::vector<float> &sintheta, const std::vector<float> &rho,
          bool debug, bool radial) {
    unsigned int nr = r.size();
    unsigned int nz = z.size();
    unsigned int ntheta = costheta.size();
    float res = 0.0;
    float thisres = 0.0;
    for (unsigned int i = 0; i < nr; i++) {
        if (radial && (r[i] > r_sampling_ii)) {
            break;
        }
        for (unsigned int j = 0; j < nz; j++) {
            for (unsigned int k = 0; k < ntheta; k++) {
                float d = pow(z[j] - z_sampling_jj, 2.0) + pow(r_sampling_ii - r[i] * sintheta[k], 2.0) +
                           r[i] * r[i] * costheta[k] * costheta[k];
                if (radial) {
                    thisres = G * rho[i] * r[i] * dv0[i] * (r_sampling_ii - r[i] * sintheta[k]) / pow(d, 1.5);
                    res += thisres;
                } else {
                    thisres = G * rho[i] * r[i] * dv0[i] * (z[j] - z_sampling_jj) / pow(d, 1.5);
                    res += thisres;
                }
                if (debug) {
                    if (i==5 && j==5 && k == 5){
                        printf("CPU \n");
                        printf("The value of f_z is %e\n", thisres);
                        printf("The value of distance is %fd\n", sqrt(d));
                        printf("The value of r[i] is %fd\n", r[i]);
                        printf("The value of z[j] is %fd\n", z[j]);
                        printf("The value of costheta is %fd\n", costheta[k]);
                        printf("The value of sintheta is %fd\n", sintheta[k]);
                        printf("The value of dv0 is %fd\n", dv0[i]);
                        printf("The value of rho is %e\n", rho[i]);
                        printf("The value of rsampling is %fd\n", r_sampling_ii);
                        printf("The value of zsampling is %fd\n", z_sampling_jj);
                        printf("The value of G is %e\n", G);
                    }
                }
            }
        }
    }
    return res;
}
// ##################################################################


std::vector<float>
get_all_g_impl_cpu(float G, const std::vector<float> &dv0, const std::vector<float> &r_sampling,
                   const std::vector<float> &z_sampling,
                   const std::vector<float> &r, const std::vector<float> &z, const std::vector<float> &costheta,
                   const std::vector<float> &sintheta, const std::vector<float> &rho,bool debug, bool radial = true) {
    std::vector<std::future<float>> futures;
    int nr = r_sampling.size();
    int nz = z_sampling.size();
    futures.reserve(nr * nz);
    // Spawn threads
    std::vector<float> f_z = zeros(nr * nz);
    for (unsigned int i = 0; i < nr; i++) {
        for (unsigned int j = 0; j < nz; j++) {
            futures.emplace_back(
                    std::async(get_g_cpu, r_sampling[i], z_sampling[j], G, dv0, r, z, costheta, sintheta, rho, debug, radial));
        }
    }

// Collect results and populate f_z
    for (unsigned int i = 0; i < nr; i++) {
        for (unsigned int j = 0; j < nz; j++) {
            f_z[i + j * nr] = futures[i * nz + j].get();
        }
    }
    return f_z;
}


std::vector<float>
get_all_g(float redshift, const std::vector<float> &dv0, const std::vector<float> &r_sampling,
          const std::vector<float> &z_sampling,
          const std::vector<float> &r, const std::vector<float> &z, const std::vector<float> &costheta,
          const std::vector<float> &sintheta, const std::vector<float> &rho, bool debug, bool radial = true, bool cuda = true) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    // G = cc.G.to(uu.km / uu.s ** 2 / uu.kg * uu.lyr ** 2).value *(1+redshift)
    float G = 7.456866768350099e-46 * (1 + redshift);
    if (device_count > 0 && cuda) {
        // If CUDA devices are available, use the CUDA implementation
        return get_all_g_impl_cuda(G, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, debug, radial);
    } else {
        // If no CUDA devices are available, use the CPU implementation
        return get_all_g_impl_cpu(G, dv0, r_sampling, z_sampling, r, z, costheta, sintheta, rho, debug, radial);
    }
}

std::vector<float> calculate_rotational_velocity(float redshift, const std::vector<float> &dv0,
                                                  std::vector<float> r_sampling,
                                                  const std::vector<float> &r,
                                                  const std::vector<float> &z,
                                                  const std::vector<float> &costheta,
                                                  const std::vector<float> &sintheta,
                                                  const std::vector<float> &rho, bool debug) {
//#######################################################################################################
//#######################################################################################################
//#######################################################################################################
//#######################################################################################################
//This function takes in two vectors: f_z, which contains the gravitational acceleration values calculated by
// the get_all_g_cuda() function, and r_sampling, which contains the radial sampling positions used in the calculation.
//
//The function first calculates the dimensions of the f_z array based on its size. Then, it allocates a result
// vector v_r with the same size as r_sampling.
//
//Next, the function calculates the velocities for each radial position. It uses the formula v**2 = f_z / r_sampling
// to calculate the squared velocity at each radial position, then takes the square root to get the actual velocity.
// This calculation is done in a loop over each position in r_sampling.
//
//Finally, the function returns the v_r vector containing the calculated velocities.
    int nr_sampling = r_sampling.size();
    float km_lyr = 9460730472580.8; //uu.lyr.to(uu.km)
    // Allocate result vector
    std::vector<float> z_sampling = {0.0};
    bool radial = true;
    bool cuda = true;
    std::vector<float> v_r(nr_sampling);
    std::vector<float> f_z = get_all_g(redshift, dv0, r_sampling, z_sampling, r, z,
                                        costheta, sintheta, rho, debug, radial, cuda);
    // Calculate velocities
    float v_squared;
    for (int i = 0; i < nr_sampling; i++) {
        v_squared = f_z[i] * r_sampling[i] * km_lyr;
        v_r[i] = sqrt(v_squared); // 9460730777119.56 km
    }
    // Return result
    return v_r;
}


std::vector<float> creategrid(float rho_0, float alpha_0, float rho_1, float alpha_1, int n) {
    if (alpha_1 > alpha_0) {
        float alpha_ = alpha_0;
        float rho_ = rho_0;
        alpha_0 = alpha_1;
        rho_0 = rho_1;
        alpha_1 = alpha_;
        rho_1 = rho_;
    }
    int n_range = 4;
    float r_max_1 = n_range / alpha_0;
    float r_max_2 = n_range / alpha_1;
    float M1 = massCalc(alpha_0, rho_0, 1.0);
    float M2 = massCalc(alpha_1, rho_1, 1.0);
    int n1 = M1 / (M1 + M2) * n;
    int n2 = M2 / (M1 + M2) * n;
    float r_min1 = 1.0;
    float r_min2 = r_max_1 + 1.0;

    // Define the grid of n points using a geometric sequence
    std::vector<float> r(n1 + n2);
    for (int i = 0; i < n1; i++) {
        r[i] = r_min1 * pow(r_max_1 / r_min1, i / (float) (n1 - 1));
    }
    for (int i = 0; i < n2; i++) {
        r[i + n1] = r_min2 * pow(r_max_2 / r_min2, i / (float) (n2 - 1));
    }
    return r;
}


class Galaxy {       // The class
public:             // Access specifier
    int nr;
    int nz;
    int nr_sampling;
    int nz_sampling;
    float R_max;
    float Mtotal_si;
    const float pi = 3.141592653589793;
    float alpha_0;
    float rho_0;
    float alpha_1;
    float rho_1;
    float h0;
    float dz;
    float redshift;
    float GalaxyMass;
    std::vector<float> r;
    std::vector<float> dv0;
    std::vector<float> z;
    std::vector<float> r_sampling;
    std::vector<float> z_sampling;
    std::vector<float> rho;
    std::vector<float> theta;
    std::vector<float> costheta;
    std::vector<float> sintheta;
    std::vector<float> f_z;
    std::vector<float> rotational_velocity_;
    // ######################################
    std::vector<float> x_rotation_points;
    int n_rotation_points = 0;
    std::vector<float> v_rotation_points;
    // ######################################
    bool radial = true;


    Galaxy(float GalaxyMass, float rho_0, float alpha_0, float rho_1, float alpha_1, float h0,
           float R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, float redshift = 0.0)
            : R_max(R_max), nr(nr), nz(nz), nr_sampling(nr_sampling), nz_sampling(nz_sampling),
              alpha_0(alpha_0), rho_0(rho_0), alpha_1(alpha_1), rho_1(rho_1), h0(h0), redshift(redshift),
              GalaxyMass(GalaxyMass) {
        // ######################################
        r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
        nr = r.size();
        z = linspace(-h0 / 2.0, h0 / 2.0, nz);
        rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        theta = linspace(0, 2 * pi, ntheta);
        costheta = costhetaFunc(theta);
        sintheta = sinthetaFunc(theta);
        // Allocate result vector
        z_sampling = linspace(-h0 / 2.0, h0 / 2.0, nz);
        r_sampling = linspace(1, R_max, nr_sampling);
        dz = h0 / nz;
        float dtheta = 2 * pi / ntheta;
        dv0.resize(1);
        dv0[0] = 0.0;
        for (int i = 1; i < nr; i++) {
            dv0.push_back((r[i] - r[i - 1]) * dz * dtheta);
        }
        int dv0size = dv0.size();
        if (dv0size != nr){
            std::cout << "error on dv0";
        }
    }


    std::vector<float> get_f_z(const std::vector<float> &x, bool debug, bool radial = true, bool cuda = true) {
        // Calculate the rotation velocity using the current values of x
        float rho_0 = x[0];
        float alpha_0 = x[1];
        float rho_1 = x[2];
        float alpha_1 = x[3];
        float h0 = x[4];
        // Calculate the total mass of the galaxy
        std::vector<float> r_sampling = this->x_rotation_points;
        std::vector<float> z_sampling={0.0};
        if(!radial){
            z_sampling = {this->h0/2.0};
        }
        std::vector<float> f_z = get_all_g(redshift, dv0, r_sampling, z_sampling, r, z,
                                            costheta, sintheta, rho, debug, radial, cuda);
        return f_z;
    }


    // Define the function to be minimized
    float error_function(const std::vector<float> &x) {
        // Calculate the rotation velocity using the current values of x
        float rho_0 = x[0];
        float alpha_0 = x[1];
        float rho_1 = x[2];
        float alpha_1 = x[3];
        float h0 = x[4];
        // Calculate the total mass of the galaxy
        float Mtotal_si = massCalc(alpha_0, rho_0, h0);  // Mtotal in Solar Masses
        float error_mass = pow((this->GalaxyMass - Mtotal_si) / this->GalaxyMass, 2);
        bool debug = false;
        std::vector<float> rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        std::vector<float> vsim = calculate_rotational_velocity(this->redshift, this->dv0,
                                                                 this->x_rotation_points,
                                                                 this->r,
                                                                 this->z,
                                                                 this->costheta,
                                                                 this->sintheta,
                                                                 rho,
                                                                 debug);
        float error = 0.0;
        for (int i = 0; i < n_rotation_points; i++) { error += pow((v_rotation_points[i] - vsim[i]), 2); }
        std::cout << "Total Error = " << (error + error_mass) << "\n";
        return error + error_mass;
    }


    void read_galaxy_rotation_curve(std::vector<std::array<float, 2>> vin) {
        n_rotation_points = vin.size();
        this->x_rotation_points.clear();
        this->v_rotation_points.clear();
        for (const auto &row: vin) {
            this->x_rotation_points.push_back(row[0]); // Extract the first column (index 0)
            this->v_rotation_points.push_back(row[1]); // Extract the first column (index 0)
        }
    }

//// Objective function for optimization
    static float objective_wrapper(const std::vector<float> &x, std::vector<float> &grad, void *data) {
        return reinterpret_cast<Galaxy *>(data)->error_function(x);
    }

    // Define the Nelder-Mead optimizer
    std::vector<double>
    nelder_mead(const std::vector<double> &x0, int max_iter = 1000, double xtol_rel = 1e-6) {
        nlopt::opt opt(nlopt::LN_NELDERMEAD, x0.size());
        opt.set_min_objective(&Galaxy::objective_wrapper, this);
        opt.set_xtol_rel(xtol_rel);
        std::vector<double> x = x0;
        double minf;
        nlopt::result result = opt.optimize(x, minf);
        if (result < 0) {
            std::cerr << "nlopt failed: " << strerror(result) << std::endl;
        }
        return x;
    }


    // End of Galaxy Class
};


int main() {
    std::vector<std::array<float, 2>> m33_rotational_curve = {
//            {0.0f,       0.0f},
//            {1508.7187f, 38.674137f},
//            {2873.3889f, 55.65067f},
//            {4116.755f,  67.91063f},
            {5451.099f,  79.22689f},
//            {6846.0957f, 85.01734f},
//            {8089.462f,  88.38242f},
//            {9393.48f,   92.42116f},
//            {10727.824f, 95.11208f},
//            {11880.212f, 98.342697f},
//            {13275.208f, 99.82048f},
//            {14609.553f, 102.10709f},
//            {18521.607f, 104.25024f},
//            {22403.336f, 107.60643f},
//            {26406.369f, 115.40966f},
//            {30379.076f, 116.87875f},
//            {34382.107f, 116.05664f},
//            {38354.813f, 117.93005f},
//            {42266.87f, 121.42091f},
//            {46300.227f, 128.55017f},
//            {50212.285f, 132.84966f}
    };


    const float M33_Distance = 3.2E6;
    const float Radius_Universe_4D = 14.03E9;
    float redshift = M33_Distance / (Radius_Universe_4D - M33_Distance);
    const int nr = 60;
    const int nz = 60;
    const int ntheta = 60;
    const int nr_sampling = 103;
    const int nz_sampling = 104;
    const float R_max = 50000.0;
    const float pi = 3.141592653589793;
    const float alpha_0 = 0.00042423668409927005;
    const float rho_0 = 12.868348904393013;
    const float alpha_1 = 2.0523892233327836e-05;
    const float rho_1 = 0.13249804158174094;
    const float h0 = 156161.88949004377;
    const float GalaxyMass = 5E10;
//    const std::vector<float> r = linspace(1,R_max, nr);
    std::vector<float> r = creategrid(rho_0, alpha_0, rho_1, alpha_1, nr);
    const std::vector<float> z = linspace(-h0 / 2.0, h0 / 2.0, nz);
    const std::vector<float> rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
    const std::vector<float> theta = linspace(0, 2 * pi, ntheta);
    std::vector<float> f_z = zeros(nr * nz);
    Galaxy M33 = Galaxy(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0,
                        R_max, nr, nz, nr_sampling, nz_sampling, ntheta,
                        redshift);
    M33.read_galaxy_rotation_curve(m33_rotational_curve);
    std::vector<float> x0 = {rho_0, alpha_0, rho_1, alpha_1, h0};
//    std::vector<float> xout = {rho_0, alpha_0, rho_1, alpha_1, h0};
//    std::vector<float> xout = {22.0752, 0.00049759, 0.122031, 1.71929e-05, 125235};
//    xout = M33.nelder_mead(x0);
//    print(xout);
    bool radial = false;
    bool debug = true;

    bool cuda = false;
    auto start = std::chrono::high_resolution_clock::now();
    f_z = M33.get_f_z(x0, debug, radial, cuda);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    std::cout << "CPU radial = " << radial << "\n";
    print(f_z);


    cuda = true;
    start = std::chrono::high_resolution_clock::now ();
    f_z = M33.get_f_z(x0, debug, radial, cuda);
    stop = std::chrono::high_resolution_clock::now ();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    std::cout << "GPU radial = " << radial << "\n";
    print(f_z);
}
