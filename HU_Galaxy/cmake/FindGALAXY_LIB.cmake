find_library(GALAXY_LIB NAMES GalaxyLibNonCuda PATHS ${CMAKE_SOURCE_DIR}/lib)
message( " where to look for GalaxyLib " ${CMAKE_SOURCE_DIR}/lib))
if(GALAXY_LIB)
    set(GALAXY_LIB_FOUND TRUE)
    message(STATUS "Found GalaxyLibNonCuda: ${GALAXY_LIB}")
else()
    set(GALAXY_LIB_FOUND FALSE)
    message(FATAL_ERROR "Could not find GalaxyLibNonCuda")
endif()

# Export variables
set(GALAXY_LIB ${GALAXY_LIB} CACHE FILEPATH "Path to GalaxyLibNonCuda library")
set(GALAXY_LIB_FOUND ${GALAXY_LIB_FOUND} CACHE BOOL "Whether GalaxyLibNonCuda library was found")
