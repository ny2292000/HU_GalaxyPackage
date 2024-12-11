cmake_minimum_required(VERSION 3.30)
project(hugalaxy_project LANGUAGES CXX)

# Check the environment variable CREATE_PACKAGE.
# If it's not set or its value is not TRUE, the condition is evaluated to FALSE.
message("CREATE_PACKAGE     $ENV{CREATE_PACKAGE} ")

if (NOT $ENV{CREATE_PACKAGE} STREQUAL "TRUE")
    message("Create testCPP")
    add_subdirectory("testCPP")
else()
    message("Create Python Package")
    add_subdirectory("./src/hugalaxy")
endif()
