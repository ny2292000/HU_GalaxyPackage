cmake_minimum_required(VERSION 3.22.0)
project(testCPPLibrary LANGUAGES CXX)


message("Torch includes - testCPPLibrary  " "${TORCH_INCLUDE_DIRS}")


add_executable(${PROJECT_NAME} TestGalaxy.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${Python_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRECTORIES}
        ${pybind11_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        ${NLopt_INCLUDE_DIRS}
        /home/mp74207/CLionProjects/HU_GalaxyPackage/HU_Galaxy
        )

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        /home/mp74207/CLionProjects/HU_GalaxyPackage/build/HU_Galaxy/libHU_Galaxy_GalaxyWrapper.so
        ${TORCH_LIBRARIES}
        ${Python_NumPy_LIBRARIES}
        ${Python_LIBRARIES}
        ${NLOPT_LIBRARIES}
        NLopt::nlopt
        pybind11::module
        pybind11::lto
        m
        c
        stdc++fs
        )
