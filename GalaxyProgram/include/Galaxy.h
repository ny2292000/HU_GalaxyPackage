//
// Created by mp74207 on 4/17/23.
//

#ifndef MAINPRONONCUDA_GALAXY_H
#define MAINPRONONCUDA_GALAXY_H

#include <vector>
#include <array>
#include <nlopt.hpp>

class Galaxy {       // The class
public:             // Access specifier
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

    Galaxy(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
           double R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta, double redshift = 0.0);

    ~Galaxy();

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    get_f_z(const std::vector<double> &x, bool debug);

    // Define the function to be minimized
    double error_function(const std::vector<double> &x);

    void read_galaxy_rotation_curve(std::vector<std::array<double, 2>> vin);

};



#endif //MAINPRONONCUDA_GALAXY_H
