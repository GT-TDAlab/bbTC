#include "bb_tc.h"

long BBTc::ListIntersectUsingHMap (Ordinal ab_src, Ordinal bc_src, Ordinal ac_src, Ordinal f_i, int worker_id) {
    long n_triangles = 0;
  
    auto hmap = hmaps_[ worker_id ];
    Ordinal b = 0;
  
    for( Ordinal i=rows_[ ac_src+f_i ]; i<rows_[ ac_src+f_i+1 ]; i++ ) {
        hmap[ cols_[i] ] = f_i;
    }
  
    for( Ordinal i=rows_[ ab_src+f_i ]; i<rows_[ ab_src+f_i+1 ]; i++ ) {
        b = cols_[ i ];
    
        for( Ordinal j=rows_[ bc_src+b ]; j<rows_[ bc_src+b+1 ]; j++ ) {
            n_triangles += ( hmap[ cols_[ j ] ] == f_i );
        }
    }
  
    return n_triangles;
}

void BBTc::TrianglesInADenseTask (Ordinal ab, Ordinal bc, Ordinal ac) {
    auto worker_id = omp_get_thread_num();
    auto hmap = hmaps_[ worker_id ];
    long n_triangles=0;
    Ordinal ab_src, bc_src, ac_src;
  
    auto ab_width = st_->GetRow(ab, ab_src);
    bc_src = tiles_[ bc ];
    ac_src = tiles_[ ac ];
  
    for( Ordinal i=0; i<st_->GetWidth( ac ); i++ ){
        hmap[ i ] = inf;
    }
  
    for( Ordinal f_i=0; f_i<ab_width; f_i++ ){  
        if( rows_[ ac_src+f_i+1 ]==rows_[ ac_src+f_i ] ){
            continue;
        }
    
        if (rows_[ab_src+f_i+1] - rows_[ab_src+f_i] < 4) {
            n_triangles += ListIntersect (ab_src, bc_src, ac_src, f_i);
        } else{
            n_triangles += ListIntersectUsingHMap( ab_src, bc_src, ac_src, f_i, worker_id );
        }
    }
  
    no_of_triangles_.fetch_add (n_triangles);
}

void BBTc::ArrangeHashMaps(){
    n_worker_ = omp_get_max_threads();
    hmaps_.resize( n_worker_ );
    for( int i=0; i<n_worker_; i++ ){
        hmaps_[ i ] = new Ordinal[ HASH_LIMIT ];
    }
}
