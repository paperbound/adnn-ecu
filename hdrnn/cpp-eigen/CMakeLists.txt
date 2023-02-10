# Set Cmake version to 3.10
cmake_minimum_required(VERSION 3.10)

# Setting up Project
project(hdrnn VERSION 0.1)

# C++ Configurations/Requirements
set(CMAKE_CXX_STANDARD, 17)
set(CMAKE_CXX_STANDARD_REQUIRED, True)

# Submodules
# TODO : add submodules update logic for hdrnn
#      	 (take inspiration from argo codebase maybe)
add_subdirectory(submodules/eigen)
add_subdirectory(submodules/gflags)

# Executable and library linkings
add_executable(hdr src/hdrnn.cc)
target_link_libraries(hdr gflags::gflags)
target_link_libraries(hdr Eigen3::Eigen)
