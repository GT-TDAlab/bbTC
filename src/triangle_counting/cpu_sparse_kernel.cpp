#include "bb_tc.h"

long BBTc::ListIntersect( Ordinal ab_src, Ordinal bc_src, Ordinal ac_src, Ordinal f_i ){
  
    Ordinal l_i, l_j, r_i, r_j, b;
  
    long n_triangles = 0;
    r_i = rows_[ ac_src+f_i+1 ];
    for( Ordinal i=rows_[ ab_src+f_i ]; i<rows_[ ab_src+f_i+1 ]; i++ ) {
        b = cols_[ i ];
    
        l_j = rows_[ bc_src+b ];
        r_j = rows_[ bc_src+b+1 ];
    
        l_i = rows_[ ac_src+f_i ];
    
        while( l_i<r_i && l_j<r_j ) {
            if( cols_[ l_i ]==cols_[ l_j ] ) {
            ++l_i;
            ++l_j;
            ++n_triangles;
            } else if( cols_[ l_i ]<cols_[ l_j ] ) {
            if( cols_[ l_j ] > cols_[ r_i-1 ] ){
                break;
            }
    
            ++l_i;
            } else {
            if( cols_[ l_i ] > cols_[ r_j-1 ] ){
                break;
            }
    
            ++l_j;
            }
        }
    }
  
    return n_triangles;
}

void BBTc::TrianglesInASparseTask( Ordinal ab, Ordinal bc, Ordinal ac ){
    Ordinal ab_src, bc_src, ac_src;
    long n_triangles=0;
    auto ab_width = st_->GetHeight( ab );
  
    ab_src = tiles_[ ab ];
    bc_src = tiles_[ bc ];
    ac_src=tiles_[ ac ];
  
    for( Ordinal f_i=0; f_i<ab_width; f_i++ ){
        if( rows_[ ac_src+f_i+1 ]==rows_[ ac_src+f_i ] ){
            continue;
        }
    
        n_triangles += ListIntersect( ab_src, bc_src, ac_src, f_i );
    }
  
    no_of_triangles_.fetch_add( n_triangles );
}
