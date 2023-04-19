#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include "../include/Galaxy.h"

using namespace boost::python;

//g++ -shared -fPIC -o my_module.so my_module.cpp -I/usr/include/python3.9 -I<path-to-numpy-headers>
// -L<path-to-boost-python-library> -lboost_python39 -lpython3.9


std::vector<double> vec_from_array(PyArrayObject *array) {
    // Check that input is a 1-dimensional array of doubles
    if (PyArray_NDIM(array) != 1 || PyArray_TYPE(array) != NPY_DOUBLE) {
        throw std::invalid_argument("Input must be a 1D NumPy array of doubles");
    }

    // Get the size of the array and a pointer to its data
    int size = PyArray_SIZE(array);
    double *data_ptr = static_cast<double *>(PyArray_DATA(array));

    // Create a vector from the array data
    std::vector<double> vec(data_ptr, data_ptr + size);

    return vec;
}

// Define a wrapper function for vec_from_array that takes a Python object as input
std::vector<double> vec_from_array_wrapper(object py_array) {
    // Convert the Python object to a NumPy array object
    PyArrayObject *array = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(py_array.ptr(), NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));

    // Call the vec_from_array function
    std::vector<double> vec = vec_from_array(array);

    // Clean up the NumPy array object
    Py_DECREF(array);

    return vec;
}

// Define a Boost.Python module
BOOST_PYTHON_MODULE(my_module) {
    // Initialize the NumPy C API
    import_array();

    // Expose the vec_from_array function to Python
    def("vec_from_array", vec_from_array_wrapper);
}


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../include/Galaxy.h"

namespace py = pybind11;

// Expose vec_from_array to Python
py::array_t<double> vec_from_array(py::array_t<double> array) {
    std::vector<double> vec = vec_from_array(reinterpret_cast<PyArrayObject *>(array.ptr()));
    return array_from_vec(vec);
}

// Expose array_from_vec to Python
py::array_t<double> array_from_vec(py::array_t<double> vec) {
    std::vector<double> vec = vec_from_array(reinterpret_cast<PyArrayObject *>(array.ptr()));
    return array_from_vec(vec);
}

// Expose your C++ functions and classes to Python
PYBIND11_MODULE(my_module, m) {
    m.doc() = "My Python module"; // Add a docstring for your module

    // Expose vec_from_array and array_from_vec to Python
    m.def("vec_from_array", vec_from_array, py::arg("array"), "Convert a NumPy array to a C++ vector");
    m.def("array_from_vec", array_from_vec, py::arg("vec"), "Convert a C++ vector to a NumPy array");

    // Expose your C++ functions and classes to Python
    py::class_<MyClass>(m, "MyClass")
            .def(py::init<>())
            .def("my_method", &MyClass::my_method);
}
