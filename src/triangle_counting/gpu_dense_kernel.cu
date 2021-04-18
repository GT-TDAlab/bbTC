#include "bb_tc_gpu.h"

__global__ void DenseKernel(
    Ordinal *ab_rows, Ordinal *ab_cols, int v_ab,
    Ordinal *bc_rows, Ordinal *bc_cols,
    Ordinal *ac_rows, Ordinal *ac_cols,
    unsigned long long int *nt,
    unsigned int* bitmap ){

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  Ordinal ab_src, bc_src, ac_src;

  ab_src = ab_rows[ 0 ];
  bc_src = bc_rows[ 0 ];
  ac_src = ac_rows[ 0 ];

  Ordinal b;
  unsigned long long int n_triangles = 0;
  unsigned int map_i, e;

  map_i = ( index*BITMAP_INT_WIDTH );

  for( Ordinal f_i=index; f_i<v_ab; f_i+=stride ){
    if( ac_rows[ f_i ] == ac_rows[ f_i+1 ] ) continue;
    for( Ordinal i=ac_rows[ f_i ]-ac_src; i<ac_rows[ f_i+1 ]-ac_src; i++ ) {
      bitmap[ map_i + ( ac_cols[ i ]>>5 ) ] |= (unsigned int)( 1<<( ac_cols[ i ]&31 ) );
    }

    for( Ordinal i=ab_rows[ f_i ]-ab_src; i<ab_rows[ f_i+1 ]-ab_src; i++ ) {
      b = ab_cols[ i ];
      for( Ordinal j=bc_rows[ b ]-bc_src; j<bc_rows[ b+1 ]-bc_src; j++ ) {
        e = bitmap[ map_i + ( bc_cols[ j ]>>5 ) ];
        if( ( e>>( ( bc_cols[ j ]&31 ) ) ) & 1 ){
          n_triangles++;
        }
      }
    }

    for( Ordinal i=ac_rows[ f_i ]-ac_src; i<ac_rows[ f_i+1 ]-ac_src; i++ ) {
      bitmap[ map_i + ( ac_cols[ i ]>>5 ) ] = 0;
    }
  }

  atomicAdd( nt, n_triangles );
}