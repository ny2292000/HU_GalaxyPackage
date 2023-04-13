
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



std::vector<double> zeros(int points);
std::vector<double> vec_from_array(PyArrayObject *array);
PyArrayObject *array_from_vec(std::vector<double> vec);
std::vector<double> costhetaFunc(const std::vector<double> &theta);
std::vector<double> sinthetaFunc(const std::vector<double> &theta);
std::vector<double> linspace(double start, double end, size_t points);
std::vector<double> density(double rho_0, double alpha_0, double rho_1, double alpha_1,std::vector<double> r );


double get_g_cpu( int ii, int jj, double G, double dv0, const std::vector<double>& r, const std::vector<double>& z,
                 const std::vector<double>& costheta, const std::vector<double>& sintheta, const std::vector<double>& rho);

std::vector<double> get_all_g_impl_cpu(int nr, int nz, double G, double dv0, const std::vector<double>& r, const std::vector<double>& z,
                                      const std::vector<double>& costheta, const std::vector<double>& sintheta, const std::vector<double>& rho);

__global__ void get_all_g_kernel(int nr, int nz, int nr_sampling, int nz_sampling, double G, double dv0,
                                 const double *r_sampling, const double *z_sampling,
                                 const double *grid_data, const double *rho,
                                 int costheta_size, bool radial, double *f_z);

std::vector<double> get_all_g_impl_cuda(double G, double dv0,
                                       const std::vector<double> &r_sampling, const std::vector<double> &z_sampling,
                                       const std::vector<double> &r, const std::vector<double> &z,
                                       const std::vector<double> &costheta, const std::vector<double> &sintheta,
                                       const std::vector<double> &rho, double redshift = 0.0, bool radial = true);

std::vector<double> get_all_g(double G, double dv0,
                             const std::vector<double> &r_sampling, const std::vector<double> &z_sampling,
                             const std::vector<double> &r, const std::vector<double> &z,
                             const std::vector<double> &costheta, const std::vector<double> &sintheta,
                             const std::vector<double> &rho, double redshift = 0.0, bool radial = true);

double massCalc(double alpha, double rho, double h);
double massCalcX(double alpha, double rho, double h, double x);


std::vector<double> calculate_rotational_velocity(double G, double dv0,
                                                 std::vector<double> r_sampling,
                                                 const std::vector<double> &r,
                                                 const std::vector<double> &z,
                                                 const std::vector<double> &costheta,
                                                 const std::vector<double> &sintheta,
                                                 const std::vector<double> &rho);


