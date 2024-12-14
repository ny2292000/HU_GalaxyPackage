#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H
#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define _USE_MATH_DEFINES
#include <Python.h>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <nlopt.hpp>
#include <future>
#include <utility>
#include "torch/torch.h"
#include "galaxy.h"
#include "./taskflow/taskflow.hpp"
#include <string>
#include <iostream>
#include "cnpy.h"

// Overload for 1D
template<typename T>
void save_npy(const std::string& filename, const std::vector<T>& arr) {
    cnpy::npy_save(filename, &arr[0], {arr.size()}, "w");
}

// Overload for 2D
template<typename T>
void save_npy(const std::string& filename, const std::vector<std::vector<T>>& arr) {
    size_t rows = arr.size();
    size_t cols = (rows > 0) ? arr[0].size() : 0;
    std::vector<T> flat_array;
    for (const auto& vec : arr) {
        flat_array.insert(flat_array.end(), vec.begin(), vec.end());
    }
    cnpy::npy_save(filename, &flat_array[0], {rows, cols}, "w");
}

//// Overload for 3D
//template<typename T>
//void save_npy(const std::string& filename, const std::vector<std::vector<std::vector<T>>>& arr) {
//    size_t depth = arr.size();
//    size_t rows = (depth > 0) ? arr[0].size() : 0;
//    size_t cols = (rows > 0 && depth > 0) ? arr[0][0].size() : 0;
//    std::vector<T> flat_array;
//    for (const auto& mat : arr) {
//        for (const auto& vec : mat) {
//            flat_array.insert(flat_array.end(), vec.begin(), vec.end());
//        }
//    }
//    cnpy::npy_save(filename, &flat_array[0], {depth, rows, cols}, "w");
//}

// Overload for 3D
template<typename T>
void save_npy(const std::string& filename, const std::vector<std::vector<std::vector<T>>>& arr) {
    size_t depth = arr.size();
    size_t rows = (depth > 0) ? arr[0].size() : 0;
    size_t cols = (rows > 0 && depth > 0) ? arr[0][0].size() : 0;
    std::vector<T> flat_array;
    for (size_t k = 0; k < depth; k++) {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                flat_array.push_back(arr[k][i][j]);
            }
        }
    }
    cnpy::npy_save(filename, &flat_array[0], {depth, rows, cols}, "w");
}

std::vector<double> geomspace(double start, double stop, int num);
bool has_nan(const std::vector<std::vector<double>>& v);
std::vector<std::array<double, 2>> interpolate(const std::vector<std::array<double, 2>>& input, size_t num_points);
std::string get_device_util(at::Tensor tensor);
void print_tensor_shape(const torch::Tensor& tensor);
void print_tensor_dimensionality(const torch::Tensor& tensor);
std::string getCudaString(bool cuda, bool taskflow);
double get_available_gpu_memory();

bool has_nan(const std::vector<std::vector<double>>& v);

std::pair<torch::Tensor, torch::Tensor> compute_chunk(
        const torch::Tensor& r_sampling,
        const torch::Tensor& z_sampling,
        const torch::Tensor& r_broadcasted,
        const torch::Tensor& dv0_broadcasted,
        const torch::Tensor& G_broadcasted,
        const torch::Tensor& rho_broadcasted,
        const torch::Tensor& sintheta_broadcasted,
        const torch::Tensor& costheta_broadcasted,
        const torch::Tensor& z_broadcasted
        );

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_torch_no_chunks(double redshift,
                        const std::vector<double> &dv0_in,
                        const std::vector<double> &r_sampling_in,
                        const std::vector<double> &z_sampling_in,
                        const std::vector<double> &r_in,
                        const std::vector<double> &z_in,
                        const std::vector<double> &costheta_in,
                        const std::vector<double> &sintheta_in,
                        const std::vector<std::vector<double>> &rho_in,
                        int GPU_ID);


std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_torch_chunks(double redshift,
                     const std::vector<double> &dv0_in,
                     const std::vector<double> &r_sampling_in,
                     const std::vector<double> &z_sampling_in,
                     const std::vector<double> &r_in,
                     const std::vector<double> &z_in,
                     const std::vector<double> &costheta_in,
                     const std::vector<double> &sintheta_in,
                     const std::vector<std::vector<double>> &rho_in,
                     int GPU_ID
);

std::pair<torch::Tensor, torch::Tensor> get_g_torch(
        double r_sampling_ii,
        double z_sampling_jj,
        const torch::Tensor& G,
        const torch::Tensor& dv0,
        const torch::Tensor& r,
        const torch::Tensor& z,
        const torch::Tensor& costheta,
        const torch::Tensor& sintheta,
        const torch::Tensor& rho
);


std::vector<double> calculate_density_parameters(double redshift);

std::vector<std::array<double, 2>> move_rotation_curve(std::vector<std::array<double, 2>>& rotation_curve, double z1 = 0.0, double z2 = 20.0);
// Returns a vector of zeros with the given size
std::vector<double> zeros_1(int size) ;

std::vector<std::vector<double>> zeros_2(int nr, int nz) ;

void print_1D(const std::vector<double> a) ;

void print_2D(const std::vector<std::vector<double>>& a);

std::vector<std::vector<double>> calculate_tau(double effective_cross_section,
                       const std::vector<std::vector<double>>& local_density, double temperature);

std::vector<double> costhetaFunc(const std::vector<double> &theta);

std::vector<double> sinthetaFunc(const std::vector<double> &theta) ;

std::vector<double> linspace(double start, double end, size_t points) ;

// # CPU functions
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_g_thread(tf::Taskflow& pool, double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
                 const std::vector<double> &z_sampling,
                 const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
                 const std::vector<double> &sintheta, const std::vector<std::vector<double>> &rho);

std::pair<double, double> get_g_cpu(double r_sampling_ii, double z_sampling_jj, double G,
                                    const std::vector<double> &dv0, const std::vector<double> &r,
                                    const std::vector<double> &z, const std::vector<double> &costheta,
                                    const std::vector<double> &sintheta, const std::vector<std::vector<double>> &rho) ;

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_all_g(double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
          const std::vector<double> &z_sampling,
          const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
          const std::vector<double> &sintheta, const std::vector<std::vector<double>> &rho) ;



#endif // TENSOR_UTILS_H
