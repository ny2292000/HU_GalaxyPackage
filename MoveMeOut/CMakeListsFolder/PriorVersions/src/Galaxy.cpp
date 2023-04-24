//
// Created by mp74207 on 4/17/23.
//
const double pi = 3.141592653589793;
#include "lib0.h"
#include "../../../../HU_Galaxy/include/Galaxy.h"

    int nr;
    int nz;
    int nr_sampling;
    int nz_sampling;
    double R_max;
    double Mtotal_si;
    double alpha_0;
    double rho_0;
    double alpha_1;
    double rho_1;
    double h0;
    double dz;
    double redshift;
    double GalaxyMass;
    std::vector<double> r;
    std::vector<double> dv0;
    std::vector<double> z;
    std::vector<double> r_sampling;
    std::vector<double> z_sampling;
    std::vector<double> rho;
    std::vector<double> theta;
    std::vector<double> costheta;
    std::vector<double> sintheta;
    std::vector<std::vector<double>> f_z;
    std::vector<double> rotational_velocity_;
    // ######################################
    std::vector<double> x_rotation_points;
    int n_rotation_points = 0;
    std::vector<double> v_rotation_points;
    // ######################################


Galaxy::Galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
       double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift)
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
    double dtheta = 2 * pi / ntheta;
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

Galaxy::~Galaxy(){};


std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>  Galaxy::get_f_z(const std::vector<double> &x, bool debug) {
    // Calculate the rotation velocity using the current values of x
    double rho_0 = x[0];
    double alpha_0 = x[1];
    double rho_1 = x[2];
    double alpha_1 = x[3];
    double h0 = x[4];
    // Calculate the total mass of the galaxy
    std::vector<double> r_sampling = this->x_rotation_points;
    std::vector<double> z_sampling = this->z;
//        if(!debug){
//            std::vector<double> z_sampling = {this->h0/2.0};
//        }else{
//            std::vector<double> z_sampling = this->z;
//        }
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> f_z = get_all_g(redshift, dv0, r_sampling, z_sampling, r, z,
                                                                                                  costheta, sintheta, rho, debug);
    return f_z;
}



void Galaxy::read_galaxy_rotation_curve(std::vector<std::array<double, 2>> vin) {
    n_rotation_points = vin.size();
    this->x_rotation_points.clear();
    this->v_rotation_points.clear();
    for (const auto &row: vin) {
        this->x_rotation_points.push_back(row[0]); // Extract the first column (index 0)
        this->v_rotation_points.push_back(row[1]); // Extract the first column (index 0)
    }
}

