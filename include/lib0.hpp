
//#include "Python.h"
//#include <vector>
//#include <thread>
//#include <memory>  // for std::unique_ptr
//#include <cmath>
//#include <stdexcept>
//#include <cstring>
//#include "numpy/arrayobject.h"
//#include "numpy/ndarraytypes.h"
//#include <array>
//#include "/usr/include/boost/python.hpp"
//#include <iostream>
//#include <future>



std::vector<float> zeros(int points);
std::vector<float> vec_from_array(PyArrayObject *array);
PyArrayObject *array_from_vec(std::vector<float> vec);
std::vector<float> costhetaFunc(const std::vector<float> &theta);
std::vector<float> sinthetaFunc(const std::vector<float> &theta);
std::vector<float> linspace(float start, float end, size_t points);
std::vector<float> density(float rho_0, float alpha_0, float rho_1, float alpha_1,std::vector<float> r );


float get_g_cpu( int ii, int jj, float G, float dv0, const std::vector<float>& r, const std::vector<float>& z,
                 const std::vector<float>& costheta, const std::vector<float>& sintheta, const std::vector<float>& rho);

std::vector<float> get_all_g_impl_cpu(int nr, int nz, float G, float dv0, const std::vector<float>& r, const std::vector<float>& z,
                                      const std::vector<float>& costheta, const std::vector<float>& sintheta, const std::vector<float>& rho);

__global__ void get_all_g_kernel(int nr, int nz, int nr_sampling, int nz_sampling, float G, float dv0,
                                 const float *r_sampling, const float *z_sampling,
                                 const float *grid_data, const float *rho,
                                 int costheta_size, bool radial, float *f_z);

std::vector<float> get_all_g_impl_cuda(float G, float dv0,
                                       const std::vector<float> &r_sampling, const std::vector<float> &z_sampling,
                                       const std::vector<float> &r, const std::vector<float> &z,
                                       const std::vector<float> &costheta, const std::vector<float> &sintheta,
                                       const std::vector<float> &rho, float redshift = 0.0, bool radial = true);

std::vector<float> get_all_g(float G, float dv0,
                             const std::vector<float> &r_sampling, const std::vector<float> &z_sampling,
                             const std::vector<float> &r, const std::vector<float> &z,
                             const std::vector<float> &costheta, const std::vector<float> &sintheta,
                             const std::vector<float> &rho, float redshift = 0.0, bool radial = true);

double massCalc(float alpha, float rho, float h, float x=0.0f);



std::vector<float> calculate_rotational_velocity(float G, float dv0,
                                                 std::vector<float> r_sampling,
                                                 const std::vector<float> &r,
                                                 const std::vector<float> &z,
                                                 const std::vector<float> &costheta,
                                                 const std::vector<float> &sintheta,
                                                 const std::vector<float> &rho,
                                                 float redshift = 0.0);


