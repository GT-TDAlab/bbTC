cmake_minimum_required(VERSION 3.10)

project(bbtc,
        LANGUAGES C CXX
)

##############################################################################

include_directories(
        PUBLIC
            include
            include/data_structures
            include/io
            include/triangle_counting
            include/utils
            src
            src/data_structures
            src/io
            src/triangle_counting
)

# find_package(SARMA REQUIRED)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

# header files 
# file(GLOB_RECURSE header_files RELATIVE_PATH include "*.h" "*.hpp")

##############################################################################
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA)
if (NOT CUDA_FOUND)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()

    add_executable(bbtc src/bbtc.cpp)

##############################################################################
else()
    enable_language(CUDA)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v \
            -ccbin=g++ \
            --default-stream per-thread -std=c++14 \
            -Xcompiler -fopenmp -O3")

    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -Xptxas -v \
            -ccbin=g++ \
            --default-stream per-thread -std=c++14 \
            -Xcompiler -fopenmp -g ")

    set(CMAKE_CUDA_ARCH 6+)
    set(CMAKE_CUDA_CODE 6+)

    set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    # Other Flags
    set(CMAKE_PREFIX_PATH "$ENV{HOME}/.local" ${CMAKE_PREFIX_PATH})
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

    add_executable(bbtc src/bbtc.cu)
endif()

target_link_libraries(bbtc stdc++fs)
