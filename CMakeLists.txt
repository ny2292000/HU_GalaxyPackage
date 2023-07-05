cmake_minimum_required(VERSION 3.22.0)
project(hugalaxy_project LANGUAGES CXX)

# Check the environment variable CREATE_PACKAGE.
# If it's not set or its value is not TRUE, the condition is evaluated to FALSE.
message("CREATE_PACKAGE     $ENV{CREATE_PACKAGE} ")
# Check if the environment variable CREATE_PACKAGE is not set or its value is not TRUE
message("FINAL CREATE_PACKAGE     $ENV{CREATE_PACKAGE} ")
if (NOT DEFINED ENV{CREATE_PACKAGE} OR NOT $ENV{CREATE_PACKAGE} STREQUAL "TRUE")
    message("Create testCPP")
    add_subdirectory("testCPP")
else()
    message("Create Python Package")
    add_subdirectory("./src/hugalaxy")
endif()
