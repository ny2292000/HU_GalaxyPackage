
std::vector<double> zeros(int points);

std::vector<double> vec_from_array(PyArrayObject *array);

PyArrayObject *array_from_vec(std::vector<double> vec);

std::vector<double> costhetaFunc(const std::vector<double> &theta);

std::vector<double> sinthetaFunc(const std::vector<double> &theta);

std::vector<double> linspace(double start, double end, size_t points);

std::vector<double> density(double rho_0, double alpha_0, double rho_1, double alpha_1, std::vector<double> r);


//std::vector<double>
//get_all_g_impl_cpu(double G, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
//                   const std::vector<double> &z_sampling,
//                   const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
//                   const std::vector<double> &sintheta, const std::vector<double> &rho, bool radial = true);
//
//__global__ void get_all_g_kernel(double G, const double* dv0, const double* r_sampling, const double* z_sampling,
//                                 const double* r, const double* z, const double* costheta, const double* sintheta,
//                                 const double* rho, int costheta_size, bool radial, double* f_z, int nr_sampling,
//                                 int nz_sampling, int nr, int nz, int ir, int iz);
//
//std::vector<double>
//get_all_g(double redshift, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
//          const std::vector<double> &z_sampling,
//          const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
//          const std::vector<double> &sintheta, const std::vector<double> &rho, bool radial = true, bool cuda = true);


double massCalc(double alpha, double rho, double h);

double massCalcX(double alpha, double rho, double h, double x);


std::vector<double> calculate_rotational_velocity(double redshift, const std::vector<double> &dv0,
                                                  std::vector<double> r_sampling,
                                                  const std::vector<double> &r,
                                                  const std::vector<double> &z,
                                                  const std::vector<double> &costheta,
                                                  const std::vector<double> &sintheta,
                                                  const std::vector<double> &rho);


