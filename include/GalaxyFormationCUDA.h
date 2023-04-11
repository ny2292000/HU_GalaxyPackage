#ifndef libGalaxyFormationCUDA_H
#define libGalaxyFormationCUDA_H
//#include "/usr/include/boost/python.hpp"
//#include "/home/mp74207/anaconda3/lib/python3.9/site-packages/numpy/core/include//numpy/arrayobject.h"


std::vector<float> get_all_g_cuda(float G, float dv0,
                                  const std::vector<float>& r_sampling, const std::vector<float>& z_sampling,
                                  const std::vector<float>& r, const std::vector<float>& z,
                                  const std::vector<float>& costheta, const std::vector<float>& sintheta,
                                  const std::vector<float>& rho, float redshift=0.0,  bool radial=true);

std::vector<float> calculate_rotational_velocity(float G, float dv0,
                                                 const std::vector<float>& r_sampling,
                                                 const std::vector<float>& r,
                                                 const std::vector<float>& z,
                                                 const std::vector<float>& costheta,
                                                 const std::vector<float>& sintheta,
                                                 const std::vector<float>& rho,
                                                 float redshift=0.0 );

//PyArrayObject* array_from_vec(std::vector<float> vec);
//
//std::vector<float> vec_from_array(PyArrayObject *array);

std::vector<float> costhetaFunc(const std::vector<float> &theta);

std::vector<float> sinthetaFunc(const std::vector<float> &theta);


std::vector<float> zeros(int points);
;

std::vector<float> linspace(float start, float end, size_t points);


std::vector<float> density(float rho_0, float alpha_0, float rho_1, float alpha_1,std::vector<float> r );


class Galaxy {       // The class
public:             // Access specifier
    int nr;
    int nz;
    int nr_sampling;
    int nz_sampling;
    float G;
    float R_max;
    const float pi = 3.141592653589793;
    float dr;
    float alpha_0;
    float rho_0;
    float alpha_1;
    float rho_1;
    float h0;
    float dz;
    float dv0;
    float redshift;
    std::vector<float> r;
    std::vector<float> z;
    std::vector<float> r_sampling;
    std::vector<float> z_sampling;
    std::vector<float> rho;
    std::vector<float> theta;
    std::vector<float> costheta;
    std::vector<float> sintheta;
    std::vector<float> f_z;
    std::vector<float> rotational_velocity;
    bool radial = true;

    Galaxy(float G, float dv0, float rho_0, float alpha_0, float rho_1, float alpha_1, float h0,
           float R_max, int nr, int nz, int nr_sampling, int nz_sampling, int ntheta,
           float redshift = 0.0) {
        // ######################################
        redshift = redshift;
        G = G*(1+redshift);
        r = linspace(1, R_max, nr);
        z = linspace(-h0 / 2.0, h0 / 2.0, nz);
        rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        theta = linspace(0, 2 * pi, ntheta);
        costheta = costhetaFunc(theta);
        sintheta = sinthetaFunc(theta);
        r = linspace(1, R_max, nr);
        z = linspace(-h0 / 2.0, h0 / 2.0, nz);
        rho = density(rho_0, alpha_0, rho_1, alpha_1, r);
        theta = linspace(0, 2 * pi, ntheta);
        sintheta = sinthetaFunc(theta);
        costheta = costhetaFunc(theta);
        dv0 = dv0;
        nr_sampling = nr_sampling;
        nz_sampling = nz_sampling;
        // Allocate result vector
        z_sampling = linspace(-h0 / 2.0, h0 / 2.0, nz);
//        z_sampling.push_back(0.0);
        r_sampling = linspace(1,R_max, nr_sampling);
        R_max = R_max;
    }

    std::vector<float> _rotational_velocity() {
        // Allocate result vector
        std::vector<float> z_sampling;
        z_sampling.push_back(0.0);
        rotational_velocity = calculate_rotational_velocity(G, dv0,
                                                            r_sampling,
                                                            r,
                                                            z,
                                                            costheta,
                                                            sintheta,
                                                            rho,
                                                            redshift);
        return rotational_velocity;
    }

    void set_radial(bool radial){
        radial = radial;
    }

    std::vector<float> get_g(bool radial){
        f_z = get_all_g_cuda(G, dv0, r_sampling, z_sampling, r, z,
                             costheta, sintheta, rho, redshift, radial);
        return f_z;
    }

};


#endif
