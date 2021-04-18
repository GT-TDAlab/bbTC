#ifndef GPU_KERNEL_
#define GPU_KERNEL_

#include <cuda.h>

#include "definitions.h"

__global__ void DenseKernel(
    Ordinal *ab_rows, Ordinal *ab_cols, int v_ab,
    Ordinal *bc_rows, Ordinal *bc_cols,
    Ordinal *ac_rows, Ordinal *ac_cols,
    unsigned long long int *nt,
    unsigned int* bitmap );


__global__ void SparseKernel(
    Ordinal *ab_rows, Ordinal *ab_cols, int v_ab,
    Ordinal *bc_rows, Ordinal *bc_cols,
    Ordinal *ac_rows, Ordinal *ac_cols,
    unsigned long long int *nt );

#include "triangle_counting/gpu_dense_kernel.cu"
#include "triangle_counting/gpu_sparse_kernel.cu"
#endif
