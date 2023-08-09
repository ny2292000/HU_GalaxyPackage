
#include <Python.h>
#include <nlopt.hpp>
#include <vector>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "hugalaxy.h"
#include "tensor_utils.h"

namespace py = pybind11;



py::array_t<double> makeNumpy_2D(const std::vector<std::vector<double>>& result) {
    // Get the dimensions of the data
    size_t nrows = result.size();
    size_t ncols = result[0].size();

    // Allocate a buffer to hold the data
    auto data = new double[nrows * ncols];
    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            data[i * ncols + j] = result[i][j];
        }
    }

    // Create a new NumPy array from the data
    auto capsule = py::capsule(data, [](void* ptr) { delete[] static_cast<double*>(ptr); });
    return py::array_t<double>({nrows, ncols}, {ncols * sizeof(double), sizeof(double)}, data, capsule);
}

py::array_t<double> makeNumpy_1D(const std::vector<double>& result) {
    // Create a new NumPy array from the data
    return py::array_t<double>(result.size(), result.data());
}

py::array_t<double> move_rotation_curve_py(const py::array_t<double>& rotation_curve_py, double z1, double z2) {
    // Convert the input py::array_t<double> to std::vector<std::array<double, 2>>
    auto rc_buf_info = rotation_curve_py.request();
    if (rc_buf_info.ndim != 2 || rc_buf_info.shape[1] != 2) {
        throw std::runtime_error("Number of dimensions must be two and the second dimension must have size 2");
    }
    std::vector<std::array<double, 2>> rotation_curve(rc_buf_info.shape[0]);
    auto* ptr = static_cast<double*>(rc_buf_info.ptr);
    for (size_t i = 0; i < rc_buf_info.shape[0]; ++i) {
        rotation_curve[i][0] = ptr[i * rc_buf_info.strides[0] / sizeof(double)];
        rotation_curve[i][1] = ptr[i * rc_buf_info.strides[0] / sizeof(double) + 1];
    }

    // Call the actual function
    std::vector<std::array<double, 2>> result = move_rotation_curve(rotation_curve, z1, z2);

    // Convert the result to py::array_t<double>
    std::vector<std::vector<double>> result_vec(result.size(), std::vector<double>(2));
    for (size_t i = 0; i < result.size(); ++i) {
        result_vec[i][0] = result[i][0];
        result_vec[i][1] = result[i][1];
    }
    return makeNumpy_2D(result_vec);
}


py::array_t<double> calculate_density_parameters_py(double redshift) {
    // Allocate a buffer to hold the data
    std::vector<double> data = calculate_density_parameters(redshift);
    auto result = makeNumpy_1D(data);
    return result;
}




GalaxyWrapper::GalaxyWrapper(double GalaxyMass, double rho_0, double alpha_0, double rho_1, double alpha_1, double h0,
              double R_max, int nr, int nz, int ntheta, double redshift, int GPU_ID, bool cuda, bool taskflow, double xtol_rel, int max_iter)
        : galaxy_(GalaxyMass, rho_0, alpha_0, rho_1, alpha_1, h0, R_max, nr, nz, ntheta, redshift, GPU_ID, cuda, taskflow, xtol_rel, max_iter) {};


py::list GalaxyWrapper::DrudePropagator(py::array_t<double>& redshifts, double deltaTime, double eta, double temperature) {
    py::buffer_info buf_info = redshifts.request();
    double* ptr = static_cast<double*>(buf_info.ptr);
    galaxy_.recalculate_masses();
    double initial_total_mass = calculate_total_mass(); // Calculate initial total mass
    std::cout << "Initial Mass: " << initial_total_mass << std::endl;
    py::list numpy_arrays;
    for (size_t i = 0; i < buf_info.size; i++) {
        double redshift = ptr[i];
        galaxy_.move_galaxy_redshift(redshift);
        std::vector<std::vector<double>> current_masses = galaxy_.DrudePropagator(redshift, deltaTime, eta, temperature);
        py::array_t<double> numpy_array_masses = makeNumpy_2D(current_masses);
        py::array_t<double> numpy_array_r = makeNumpy_1D(galaxy_.r);
        py::array_t<double> numpy_array_dv0 = makeNumpy_1D(galaxy_.dv0);
        py::array_t<double> numpy_array_z = makeNumpy_1D(galaxy_.z);
        numpy_arrays.append(std::make_tuple(numpy_array_masses, numpy_array_r, numpy_array_dv0, numpy_array_z));
    }
    return numpy_arrays;
}

std::vector<double> to_std_vector(const pybind11::array_t<double>& input)
{
    std::vector<double> output;
    output.reserve(input.size());

    for (size_t i = 0; i < input.size(); i++)
    {
        output.push_back(*input.data(i));
    }
    return output;
}

void GalaxyWrapper::DrudeGalaxyFormation(py::array_t<double> &epochs, py::array_t<double> &redshifts, double eta,
                                         double temperature, const py::str &filename) {
    std::vector<double> epochs_vec = to_std_vector(epochs);
    std::vector<double> redshifts_vec = to_std_vector(redshifts);
    galaxy_.DrudeGalaxyFormation(epochs_vec, redshifts_vec, eta, temperature, filename);
}


py::array_t<double> GalaxyWrapper::density_wrapper(double rho_0, double alpha_0, double rho_1, double alpha_1, const py::array_t<double>& r, const py::array_t<double>& z) const {
    // Convert py::array_t to std::vector
    std::vector<double> r_vec(r.size());
    std::vector<double> z_vec(z.size());
    std::memcpy(r_vec.data(), r.data(), r.size() * sizeof(double));
    std::memcpy(z_vec.data(), z.data(), z.size() * sizeof(double));
    const galaxy& g = get_galaxy();
    auto density = g.density(rho_0, alpha_0, rho_1, alpha_1, r_vec,z_vec);
    auto result = makeNumpy_2D(density);
    return result;
}

py::array_t<double> GalaxyWrapper::density_wrapper_internal()  {
    const galaxy& g = get_galaxy();
    auto density = g.density(g.rho_0, g.alpha_0, g.rho_1, g.alpha_1, g.r,g.z);
    auto result = makeNumpy_2D(density);
    return result;
}

std::pair<py::array_t<double>, py::array_t<double>> GalaxyWrapper::get_f_z(const std::vector<std::vector<double>> &rho_, bool calc_vel,  const double height) {
        auto f_z_pair = galaxy_.get_f_z(rho_, calc_vel, height);

        // Get the dimensions of the f_z_r and f_z_z arrays
        size_t rows = f_z_pair.first.size();
        size_t cols = f_z_pair.first[0].size();

        // Flatten the 2D f_z_r and f_z_z vectors
        std::vector<double> f_z_r_flat;
        std::vector<double> f_z_z_flat;
        for (size_t i = 0; i < rows; ++i) {
            f_z_r_flat.insert(f_z_r_flat.end(), f_z_pair.first[i].begin(), f_z_pair.first[i].end());
            f_z_z_flat.insert(f_z_z_flat.end(), f_z_pair.second[i].begin(), f_z_pair.second[i].end());
        }

        // Convert the result to NumPy arrays
        py::array_t<double> f_z_r({rows, cols}, f_z_r_flat.data());
        py::array_t<double> f_z_z({rows, cols}, f_z_z_flat.data());

        return std::make_pair(f_z_r, f_z_z);
    };

void GalaxyWrapper::read_galaxy_rotation_curve(py::array_t<double, py::array::c_style | py::array::forcecast> vin) {
    auto buf = vin.request();
    if (buf.ndim != 2 || buf.shape[1] != 2)
        throw std::runtime_error("Input should be a 2D array with 2 columns");

    std::vector<std::array<double, 2>> vec(buf.shape[0]);
    auto ptr = static_cast<double *>(buf.ptr);
    for (ssize_t i = 0; i < buf.shape[0]; ++i) {
        vec[i][0] = ptr[i * buf.shape[1]];
        vec[i][1] = ptr[i * buf.shape[1] + 1];
    }
    galaxy_.read_galaxy_rotation_curve(vec);
}

double GalaxyWrapper::calculate_mass(double rho, double alpha, double h) {
    auto result = galaxy_.calculate_mass(rho, alpha, h);
    return result;
}



py::list GalaxyWrapper::print_density_parameters() {
    py::list density_params;
    density_params.append(galaxy_.rho_0);
    density_params.append(galaxy_.alpha_0);
    density_params.append(galaxy_.rho_1);
    density_params.append(galaxy_.alpha_1);
    density_params.append(galaxy_.h0);
    return density_params;
}

py::array_t<double> GalaxyWrapper::simulate_rotation_curve() {
    // Get the galaxy_ object

    // Calculate density at all radii
    const galaxy& g = get_galaxy();
    std::string compute_choice = getCudaString(g.cuda, g.taskflow_);
    std::cout << "Compute Choice " << compute_choice << std::endl;
    std::vector<double> xout = galaxy_.simulate_rotation_curve();

    // Convert the result to a NumPy array
    double *data = xout.data(); // Get a pointer to the underlying data
    std::size_t size = xout.size(); // Get the size of the vector
    py::array_t<double> result(size, data);

    return result;
}

// Getter and setter for v_simulated_points vector
py::array_t<double> GalaxyWrapper::get_v_simulated_points() const {
    const double* data_ptr = galaxy_.v_simulated_points.data();
    py::array_t<double> arr({static_cast<ssize_t>(galaxy_.v_simulated_points.size())}, data_ptr);
    return arr;
}
void GalaxyWrapper::set_v_simulated_points(const py::array_t<double>& arr) {
    auto buf = arr.unchecked<1>(); // Extract the underlying buffer of arr as a 1D array
    unsigned int size = buf.shape(0);
    galaxy_.v_simulated_points.resize(size);
    for (unsigned int i = 0; i < size; i++) {
        galaxy_.v_simulated_points[i] = buf(i);
    }
}




// Getter and setter for r vector
py::array_t<double> GalaxyWrapper::get_r() const {
    const double* data_ptr = galaxy_.r.data();
    py::array_t<double> arr({static_cast<ssize_t>(galaxy_.r.size())}, data_ptr);
    return arr;
}
void GalaxyWrapper::set_r(const py::array_t<double>& arr) {
    auto buf = arr.unchecked<1>(); // Extract the underlying buffer of arr as a 1D array
    unsigned int size = buf.shape(0);
    galaxy_.r.resize(size);
    for (unsigned int i = 0; i < size; i++) {
        galaxy_.r[i] = buf(i);
    }
}


// Getter and setter for r vector
py::array_t<double> GalaxyWrapper::get_z() const {
    const double* data_ptr = galaxy_.z.data();
    py::array_t<double> arr({static_cast<ssize_t>(galaxy_.z.size())}, data_ptr);
    return arr;
}
void GalaxyWrapper::set_z(const py::array_t<double>& arr) {
    auto buf = arr.unchecked<1>(); // Extract the underlying buffer of arr as a 1D array
    unsigned int size = buf.shape(0);
    galaxy_.z.resize(size);
    for (unsigned int i = 0; i < size; i++) {
        galaxy_.z[i] = buf(i);
    }
}


// Getter and setter for r vector
py::array_t<double> GalaxyWrapper::get_costheta() const {
    const double* data_ptr = galaxy_.costheta.data();
    py::array_t<double> arr({static_cast<ssize_t>(galaxy_.costheta.size())}, data_ptr);
    return arr;
}
void GalaxyWrapper::set_costheta(const py::array_t<double>& arr) {
    auto buf = arr.unchecked<1>(); // Extract the underlying buffer of arr as a 1D array
    unsigned int size = buf.shape(0);
    galaxy_.costheta.resize(size);
    for (unsigned int i = 0; i < size; i++) {
        galaxy_.costheta[i] = buf(i);
    }
}


// Getter and setter for r vector
py::array_t<double> GalaxyWrapper::get_x_rotation_points() const {
    const double* data_ptr = galaxy_.x_rotation_points.data();
    py::array_t<double> arr({static_cast<ssize_t>(galaxy_.x_rotation_points.size())}, data_ptr);
    return arr;
}
void GalaxyWrapper::set_x_rotation_points(const py::array_t<double>& arr) {
    auto buf = arr.unchecked<1>(); // Extract the underlying buffer of arr as a 1D array
    unsigned int size = buf.shape(0);
    galaxy_.x_rotation_points.resize(size);
    for (unsigned int i = 0; i < size; i++) {
        galaxy_.x_rotation_points[i] = buf(i);
    }
}

py::array_t<double> GalaxyWrapper::get_v_rotation_points() const {
    const double* data_ptr = galaxy_.v_rotation_points.data();
    py::array_t<double> arr({static_cast<ssize_t>(galaxy_.v_rotation_points.size())}, data_ptr);
    return arr;
}
void GalaxyWrapper::set_v_rotation_points(const py::array_t<double>& arr) {
    auto buf = arr.unchecked<1>(); // Extract the underlying buffer of arr as a 1D array
    unsigned int size = buf.shape(0);
    galaxy_.v_rotation_points.resize(size);
    for (unsigned int i = 0; i < size; i++) {
        galaxy_.v_rotation_points[i] = buf(i);
    }
}



// Getter and setter for r vector
py::array_t<double> GalaxyWrapper::get_sintheta() const {
    const double* data_ptr = galaxy_.sintheta.data();
    py::array_t<double> arr({static_cast<ssize_t>(galaxy_.sintheta.size())}, data_ptr);
    return arr;
}
void GalaxyWrapper::set_sintheta(const py::array_t<double>& arr) {
    auto buf = arr.unchecked<1>(); // Extract the underlying buffer of arr as a 1D array
    unsigned int size = buf.shape(0);
    galaxy_.sintheta.resize(size);
    for (unsigned int i = 0; i < size; i++) {
        galaxy_.sintheta[i] = buf(i);
    }
}




// Getter and setter for r vector
py::array_t<double> GalaxyWrapper::get_dv0() const {
    const double* data_ptr = galaxy_.dv0.data();
    py::array_t<double> arr({static_cast<ssize_t>(galaxy_.dv0.size())}, data_ptr);
    return arr;
}
void GalaxyWrapper::set_dv0(const py::array_t<double>& arr) {
    auto buf = arr.unchecked<1>(); // Extract the underlying buffer of arr as a 1D array
    unsigned int size = buf.shape(0);
    galaxy_.dv0.resize(size);
    for (unsigned int i = 0; i < size; i++) {
        galaxy_.dv0[i] = buf(i);
    }
}

// Getter and setter for rotation_curve
py::array_t<double> GalaxyWrapper::get_rotation_curve() const {
    const double* data_ptr_x = galaxy_.x_rotation_points.data();
    py::array_t<double> arrx({static_cast<ssize_t>(galaxy_.x_rotation_points.size())}, data_ptr_x);

    const double* data_ptr_v = galaxy_.v_rotation_points.data();
    py::array_t<double> arrv({static_cast<ssize_t>(galaxy_.v_rotation_points.size())}, data_ptr_v);

    // Create a tuple of arrx and arrv
    py::tuple result(2);
    result[0] = arrx;
    result[1] = arrv;

    return result;
}

void GalaxyWrapper::set_rotation_curve(const py::array_t<double>& arr) {
    auto buf = arr.unchecked<2>(); // Extract the underlying buffer of arr as a 2D array
    unsigned int size = buf.shape(1);
    galaxy_.x_rotation_points.resize(size);
    galaxy_.v_rotation_points.resize(size);
    for (unsigned int i = 0; i < size; i++) {
        galaxy_.x_rotation_points[i] = buf(0, i);
        galaxy_.v_rotation_points[i] = buf(1, i);
    }
}


void GalaxyWrapper::move_galaxy_redshift (double redshift) {
    galaxy_.move_galaxy_redshift(redshift);
}

const py::array_t<double> GalaxyWrapper::calculate_rotational_velocity (py::array_t<double> rho_py, double height)  {
    if (rho_py.ndim() != 2) {
        throw std::runtime_error("Number of dimensions for rho must be two");
    }

// convert 2d numpy array to 2d vector
    py::buffer_info info = rho_py.request();
    auto ptr = static_cast<double*>(info.ptr);
    int X = info.shape[0];
    int Y = info.shape[1];

    std::vector<std::vector<double>> rho_vec;
    rho_vec.reserve(X);
    for (int i = 0; i < X; ++i) {
        rho_vec.push_back(std::vector<double>(ptr + (i*Y), ptr + ((i+1)*Y)));
    }

// access the galaxy instance from the wrapper
    const galaxy& g = get_galaxy();
    std::vector<double> v_r = g.calculate_rotational_velocity(rho_vec, height);
    return makeNumpy_1D(v_r);
}


const py::array_t<double> GalaxyWrapper::calculate_rotational_velocity_internal ()  {
// access the galaxy instance from the wrapper
    const galaxy& g = get_galaxy();
    std::vector<double> v_r = g.calculate_rotational_velocity(g.rho);
    return makeNumpy_1D(v_r);
}

py::array_t<double> GalaxyWrapper::get_rho_py() const {
    const galaxy& g = get_galaxy();
    return makeNumpy_2D(g.rho);
}

void GalaxyWrapper::set_rho_py(const py::array_t<double>& rho_py) {
    // Convert the input py::array_t<double> to std::vector<std::vector<double>>
    auto rho_buf_info = rho_py.request();
    if (rho_buf_info.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be two");
    }
    galaxy& g = get_galaxy();
    g.rho.resize(rho_buf_info.shape[0]);
    double* ptr = static_cast<double*>(rho_buf_info.ptr);
    for (size_t i = 0; i < rho_buf_info.shape[0]; ++i) {
        g.rho[i].resize(rho_buf_info.shape[1]);
        for (size_t j = 0; j < rho_buf_info.shape[1]; ++j) {
            g.rho[i][j] = ptr[i * rho_buf_info.strides[0] / sizeof(double) + j];
        }
    }
}


// Setter member functions
void GalaxyWrapper::set_redshift(double redshift) { galaxy_.redshift = redshift; }
void GalaxyWrapper::set_R_max(double R_max) { galaxy_.R_max = R_max; }
void GalaxyWrapper::set_nz(int nz) { galaxy_.nz = nz;}
void GalaxyWrapper::set_nr(int nr) { galaxy_.nr = nr;}
void GalaxyWrapper::set_alpha_0(double alpha_0) { galaxy_.alpha_0 = alpha_0;}
void GalaxyWrapper::set_alpha_1(double alpha_1) { galaxy_.alpha_1 = alpha_1;}
void GalaxyWrapper::set_rho_0(double rho_0) { galaxy_.rho_0 = rho_0;}
void GalaxyWrapper::set_rho_1(double rho_1) { galaxy_.rho_1 = rho_1;}
void GalaxyWrapper::set_h0(double h0) { galaxy_.h0 = h0;}
void GalaxyWrapper::set_cuda(bool value) {galaxy_.cuda=value;};
void GalaxyWrapper::set_taskflow(bool value) {galaxy_.taskflow_=value;};
void GalaxyWrapper::set_GPU_ID(int value) {galaxy_.GPU_ID=value;};
void GalaxyWrapper::set_xtol_rel(double value) {galaxy_.xtol_rel=value;};
void GalaxyWrapper::set_max_iter(int value) {galaxy_.max_iter=value;};

// Getter member functions
double GalaxyWrapper::get_redshift() const { return galaxy_.redshift; }
double GalaxyWrapper::get_R_max() const { return galaxy_.R_max; }
int GalaxyWrapper::get_nz() const { return galaxy_.nz; }
int GalaxyWrapper::get_nr() const { return galaxy_.nr; }
double GalaxyWrapper::get_alpha_0() const { return galaxy_.alpha_0; }
double GalaxyWrapper::get_alpha_1() const { return galaxy_.alpha_1; }
double GalaxyWrapper::get_rho_0() const { return galaxy_.rho_0; }
double GalaxyWrapper::get_rho_1() const { return galaxy_.rho_1; }
double GalaxyWrapper::get_h0() const { return galaxy_.h0; }
const galaxy& GalaxyWrapper::get_galaxy() const { return galaxy_; }
galaxy& GalaxyWrapper::get_galaxy() { return galaxy_; }
bool GalaxyWrapper::get_cuda() const {return galaxy_.cuda;};
bool GalaxyWrapper::get_taskflow() const {return galaxy_.taskflow_;};
int GalaxyWrapper::get_GPU_ID() const {return galaxy_.GPU_ID;}
double GalaxyWrapper::get_xtol_rel() const {return galaxy_.xtol_rel;}
int GalaxyWrapper::get_max_iter() const {return galaxy_.max_iter;}

double GalaxyWrapper::calculate_total_mass() {
    double totalmass = galaxy_.calculate_total_mass();
    return totalmass;
}




PYBIND11_MODULE(hugalaxy, m) {
    py::class_<GalaxyWrapper>(m, "GalaxyWrapper")
            .def(py::init<double, double, double, double, double, double, double, int, int, int, double,int, bool, bool, double, int>(),
                 py::arg("GalaxyMass"), py::arg("rho_0"), py::arg("alpha_0"), py::arg("rho_1"), py::arg("alpha_1"), py::arg("h0"),
                 py::arg("R_max"), py::arg("nr"), py::arg("nz"), py::arg("ntheta"), py::arg("redshift") = 0.0, py::arg("GPU_ID") = 0, py::arg("cuda") = false, py::arg("taskflow") = false, py::arg("xtol_rel")=1E-6, py::arg("max_iter")=5000)
            .def("calculate_rotational_velocity", &GalaxyWrapper::calculate_rotational_velocity, py::arg("rho_py"), py::arg("height"))
            .def("DrudeGalaxyFormation", &GalaxyWrapper::DrudeGalaxyFormation, py::arg("epochs"), py::arg("redshifts"), py::arg("eta"), py::arg("temperature"), py::arg("filename"))
            .def("calculate_rotation_velocity_internal", &GalaxyWrapper::calculate_rotational_velocity_internal)
            .def("DrudePropagator", &GalaxyWrapper::DrudePropagator, py::arg("redshifts"), py::arg("time_step_years"), py::arg("eta"), py::arg("temperature"),
                 "Propagate the mass distribution in a galaxy_ using the Drude model")
            .def("get_galaxy", static_cast<const galaxy& (GalaxyWrapper::*)() const>(&GalaxyWrapper::get_galaxy))
            .def("get_galaxy_mutable", static_cast<galaxy& (GalaxyWrapper::*)()>(&GalaxyWrapper::get_galaxy))
            .def("calculate_mass", &GalaxyWrapper::calculate_mass, py::arg("rho"), py::arg("alpha"), py::arg("h0"), "A function to calculate the mass of the galaxy_")
            .def("calculate_total_mass", &GalaxyWrapper::calculate_total_mass, "A function to calculate the total mass of the galaxy_")
            .def_property("redshift", &GalaxyWrapper::get_redshift, &GalaxyWrapper::set_redshift)
            .def_property("R_max", &GalaxyWrapper::get_R_max, &GalaxyWrapper::set_R_max)
            .def_property("nz", &GalaxyWrapper::get_nz, &GalaxyWrapper::set_nz)
            .def_property("nr", &GalaxyWrapper::get_nr, &GalaxyWrapper::set_nr)
            .def_property("alpha_0", &GalaxyWrapper::get_alpha_0, &GalaxyWrapper::set_alpha_0)
            .def_property("alpha_1", &GalaxyWrapper::get_alpha_1, &GalaxyWrapper::set_alpha_1)
            .def_property("rho_0", &GalaxyWrapper::get_rho_0, &GalaxyWrapper::set_rho_0)
            .def_property("rho_1", &GalaxyWrapper::get_rho_1, &GalaxyWrapper::set_rho_1)
            .def_property("h0", &GalaxyWrapper::get_h0, &GalaxyWrapper::set_h0)
            .def_property("cuda", &GalaxyWrapper::get_cuda, &GalaxyWrapper::set_cuda)
            .def_property("taskflow", &GalaxyWrapper::get_taskflow, &GalaxyWrapper::set_taskflow)
            .def_property("rho", &GalaxyWrapper::get_rho_py, &GalaxyWrapper::set_rho_py)
            .def_property("GPU_ID", &GalaxyWrapper::get_GPU_ID, &GalaxyWrapper::set_GPU_ID)
            .def_property("max_iter", &GalaxyWrapper::get_max_iter, &GalaxyWrapper::set_max_iter)
            .def_property("xtol_rel", &GalaxyWrapper::get_xtol_rel, &GalaxyWrapper::set_xtol_rel)
            .def("move_galaxy_redshift", &GalaxyWrapper::move_galaxy_redshift, py::arg("redshift"), "Recalculate new density parameters and recreate grid")
            .def("read_galaxy_rotation_curve", &GalaxyWrapper::read_galaxy_rotation_curve)
            .def("get_f_z", &GalaxyWrapper::get_f_z, py::arg("rho"), py::arg("calc_vel"), py::arg("height")  )
            .def("simulate_rotation_curve", &GalaxyWrapper::simulate_rotation_curve)
            .def("print_density_parameters", &GalaxyWrapper::print_density_parameters)
            .def_property("r", &GalaxyWrapper::get_r, &GalaxyWrapper::set_r)
            .def_property("z", &GalaxyWrapper::get_z, &GalaxyWrapper::set_z)
            .def_property("x_rotation_points", &GalaxyWrapper::get_x_rotation_points, &GalaxyWrapper::set_x_rotation_points)
            .def_property("v_rotation_points", &GalaxyWrapper::get_v_rotation_points, &GalaxyWrapper::set_v_rotation_points)
            .def_property("v_simulated_points", &GalaxyWrapper::get_v_simulated_points, &GalaxyWrapper::set_v_simulated_points)
            .def_property("costheta", &GalaxyWrapper::get_costheta, &GalaxyWrapper::set_costheta)
            .def_property("sintheta", &GalaxyWrapper::get_sintheta, &GalaxyWrapper::set_sintheta)
            .def_property("dv0", &GalaxyWrapper::get_dv0, &GalaxyWrapper::set_dv0)
            .def_property("rotation_curve", &GalaxyWrapper::get_rotation_curve, &GalaxyWrapper::set_rotation_curve)
            .def("density_wrapper_internal", &GalaxyWrapper::density_wrapper_internal, "Calculate density using internal parameters")
            .def("density_wrapper", &GalaxyWrapper::density_wrapper, py::arg("rho_0"), py::arg("alpha_0"), py::arg("rho_1"), py::arg("alpha_1"), py::arg("r"), py::arg("z"), "Calculate density using the given parameters");
    m.def("calculate_density_parameters", &calculate_density_parameters_py, py::arg("redshift"));
    m.def("move_rotation_curve", &move_rotation_curve_py, py::arg("rotation_curve"), py::arg("z1"), py::arg("z2"));

}


