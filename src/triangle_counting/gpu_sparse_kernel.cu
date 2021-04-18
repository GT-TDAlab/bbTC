#include <cuda.h>

#include "bb_tc_gpu.h"
#include "definitions.h"

__global__ void SparseKernel(
    Ordinal *ab_rows, Ordinal *ab_cols, int v_ab,
    Ordinal *bc_rows, Ordinal *bc_cols,
    Ordinal *ac_rows, Ordinal *ac_cols,
    unsigned long long int *nt ){

  // printf("%d %d %d \n", v_ab, v_bc, v_ac );
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  Ordinal ab_src, bc_src, ac_src;

  ab_src = ab_rows[ 0 ];
  bc_src = bc_rows[ 0 ];
  ac_src = ac_rows[ 0 ];

  Ordinal l_i, l_j, r_i, r_j, b;
  unsigned long long int n_triangles = 0;

  for( Ordinal f_i=index; f_i<v_ab; f_i+=stride ){
    r_i = ac_rows[ f_i+1 ]-ac_src;
    if( r_i==(ac_rows[ f_i ]-ac_src) ) continue;
    for( Ordinal i=ab_rows[ f_i ]-ab_src; i<ab_rows[ f_i+1 ]-ab_src; i++ ) {
      b = ab_cols[ i ];
      l_j = bc_rows[ b ]-bc_src;
      r_j = bc_rows[ b+1 ]-bc_src;

      l_i = ac_rows[ f_i ]-ac_src;
      while( l_i<r_i && l_j<r_j ) {
        if( ac_cols[ l_i ]==bc_cols[ l_j ] ) {
          ++l_i;
          ++l_j;
          ++n_triangles;
        } else if( ac_cols[ l_i ]<bc_cols[ l_j ] ) {
          ++l_i;
        } else {
          ++l_j;
        }
      }
    }
  }

  atomicAdd( nt, n_triangles );
}
