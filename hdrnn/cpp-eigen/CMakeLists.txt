# Set Cmake version to 3.10
cmake_minimum_required(VERSION 3.10)

# Setting up Project
project(hdrnn VERSION 0.1)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# C++ Configurations/Requirements
set(CMAKE_CXX_STANDARD, 11)
set(CMAKE_CXX_STANDARD_REQUIRED, True)

# Submodules
# TODO : add submodules update logic for hdrnn
#      	 (take inspiration from argo codebase maybe)
add_subdirectory(submodules/eigen)
add_subdirectory(submodules/gflags)

# Executable properties and library linkings
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_executable(hdr src/hdrnn.cc)

target_link_libraries(hdr gflags::gflags)
target_link_libraries(hdr Eigen3::Eigen)
