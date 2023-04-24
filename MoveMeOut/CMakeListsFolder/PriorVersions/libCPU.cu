//double
//get_g_cpu(double r_sampling_ii, double z_sampling_jj, double G, const std::vector<double> &dv0,
//          const std::vector<double> &r,
//          const std::vector<double> &z,
//          const std::vector<double> &costheta, const std::vector<double> &sintheta, const std::vector<double> &rho,
//          bool debug, bool radial) {
//    unsigned int nr = r.size();
//    unsigned int nz = z.size();
//    unsigned int ntheta = costheta.size();
//    double res = 0.0;
//    double thisres = 0.0;
//    for (unsigned int i = 0; i < nr; i++) {
//        if (radial && (r[i] > r_sampling_ii)) {
//            break;
//        }
//        for (unsigned int j = 0; j < nz; j++) {
//            for (unsigned int k = 0; k < ntheta; k++) {
//                double d = (z[j] - z_sampling_jj, 2.0) + (r_sampling_ii - r[i] * sintheta[k], 2.0) +
//                           r[i] * r[i] * costheta[k] * costheta[k];
//                if (radial) {
//                    thisres = G * rho[i] * r[i] * dv0[i] * (r_sampling_ii - r[i] * sintheta[k]) / (d, 1.5);
//                    res += thisres;
//                } else {
//                    thisres = G * rho[i] * r[i] * dv0[i] * (z[j] - z_sampling_jj) / (d, 1.5);
//                    res += thisres;
//                }
//                if (debug) {
//                    if (i==5 && j==5 && k == 5){
//                        printf("CPU \n");
//                        printf("The value of f_z is %e\n", thisres);
//                        printf("The value of distance is %fd\n", sqrt(d));
//                        printf("The value of r[i] is %fd\n", r[i]);
//                        printf("The value of z[j] is %fd\n", z[j]);
//                        printf("The value of costheta is %fd\n", costheta[k]);
//                        printf("The value of sintheta is %fd\n", sintheta[k]);
//                        printf("The value of dv0 is %fd\n", dv0[i]);
//                        printf("The value of rho is %e\n", rho[i]);
//                        printf("The value of rsampling is %fd\n", r_sampling_ii);
//                        printf("The value of zsampling is %fd\n", z_sampling_jj);
//                        printf("The value of G is %e\n", G);
//                    }
//                }
//            }
//        }
//    }
//    return res;
//}
//// ##################################################################


//std::vector<double>
//get_all_g_impl_cpu(double G, const std::vector<double> &dv0, const std::vector<double> &r_sampling,
//                   const std::vector<double> &z_sampling,
//                   const std::vector<double> &r, const std::vector<double> &z, const std::vector<double> &costheta,
//                   const std::vector<double> &sintheta, const std::vector<double> &rho,bool debug, bool radial = true) {
//    std::vector<std::future<double>> futures;
//    int nr = r_sampling.size();
//    int nz = z_sampling.size();
//    futures.reserve(nr * nz);
//    // Spawn threads
//    std::vector<double> f_z = zeros(nr * nz);
//    for (unsigned int i = 0; i < nr; i++) {
//        for (unsigned int j = 0; j < nz; j++) {
//            futures.emplace_back(
//                    std::async(get_g_cpu, r_sampling[i], z_sampling[j], G, dv0, r, z, costheta, sintheta, rho, debug, radial));
//        }
//    }
//
//// Collect results and populate f_z
//    for (unsigned int i = 0; i < nr; i++) {
//        for (unsigned int j = 0; j < nz; j++) {
//            f_z[i + j * nr] = futures[i * nz + j].get();
//        }
//    }
//    return f_z;
//}