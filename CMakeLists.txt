cmake_minimum_required(VERSION 3.18)

project(cuda_efficient_features)

option(BUILD_SAMPLES "Build samples" OFF)
option(BUILD_TESTS   "Build tests" OFF)
set(CUDA_ARCHS "52;61;72;75;86" CACHE STRING "List of architectures to generate device code for")

# ----- Set build type -----

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(BUILD_SHARED_LIBS "Build cuda_efficient_features as a shared library" ON)

include(CMakePackageConfigHelpers)

# Generate cmake configuration scripts
set(CUDA_EFFICIENT_FEATURES_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(CUDA_EFFICIENT_FEATURES_VERSION_CONFIG ${CUDA_EFFICIENT_FEATURES_GENERATED_DIR}/${PROJECT_NAME}ConfigVersion.cmake)
set(CUDA_EFFICIENT_FEATURES_PROJECT_CONFIG ${CUDA_EFFICIENT_FEATURES_GENERATED_DIR}/${PROJECT_NAME}Config.cmake)
set(CUDA_EFFICIENT_FEATURES_TARGETS_EXPORT_NAME ${PROJECT_NAME}Targets)
set(CUDA_EFFICIENT_FEATURES_CONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME})
set(CUDA_EFFICIENT_FEATURES_NAMESPACE "${PROJECT_NAME}::")
set(CUDA_EFFICIENT_FEATURES_VERSION 1.0.0)

# Create a version config file
write_basic_package_version_file(${CUDA_EFFICIENT_FEATURES_VERSION_CONFIG}
                                 VERSION ${CUDA_EFFICIENT_FEATURES_VERSION}
                                 COMPATIBILITY SameMajorVersion)
# Create a project config file
configure_file(${PROJECT_SOURCE_DIR}/cmake/Config.cmake.in ${CUDA_EFFICIENT_FEATURES_PROJECT_CONFIG} @ONLY)

# Install to the specified directory
install(FILES ${CUDA_EFFICIENT_FEATURES_VERSION_CONFIG} ${CUDA_EFFICIENT_FEATURES_PROJECT_CONFIG}
        DESTINATION ${CUDA_EFFICIENT_FEATURES_CONFIG_INSTALL_DIR})
install(EXPORT ${CUDA_EFFICIENT_FEATURES_TARGETS_EXPORT_NAME}
        NAMESPACE ${CUDA_EFFICIENT_FEATURES_NAMESPACE}
        DESTINATION ${CUDA_EFFICIENT_FEATURES_CONFIG_INSTALL_DIR})

# Set standard installation directories
set(RUNTIME_DESTINATION bin)
set(LIBRARY_DESTINATION lib)
set(ARCHIVE_DESTINATION lib)
set(INCLUDES_DESTINATION include)
set(CUDA_EFFICIENT_FEATURES_INCLUDE_INSTALL_DIR ${INCLUDES_DESTINATION}/cuda_efficient_features)

# ----- Build -----
add_compile_options(-g) #GDB

add_subdirectory(modules)
if(BUILD_SAMPLES)
	add_subdirectory(samples)
endif()
if(BUILD_TESTS)
	add_subdirectory(tests)
endif()
