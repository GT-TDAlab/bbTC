#define ENABLE_CUDA

#include <cmath>
#include <cuda.h>

#include "bb_tc.h"

#include "block_csr.h"

#include "definitions.h"

#include "utils.hpp"
#include "tasks.hpp"

int main (int argc, const char** argv) {

    utils::Parameters params;
    if (utils::ParseArguments(params, argc, argv) < 0) {
        std::cout << "Wrong input" << std::endl;
        std::exit(0);
    }

    if (!utils::FileExist(params.file_name)){
        std::cout << "File not found." << std::endl;
        std::exit (0);
    }

    BlockCSR<Ordinal, Vertex> st = utils::BlockGraph<Ordinal, Vertex, double> (
        params.file_name,
        params.part_name,
        params.alpha,
        params.ncut
    );

    Ordinal P = st.GetNoOfCuts();

    auto t_list = tasks::TaskList(P);
    tasks::SortTasks( t_list, st);

    BBTc bbtc( &st, t_list, P );
    bbtc.InitFlags();
    bbtc.SetNGPU( params.n_gpu );

    std::cout << "=====================================" << std::endl;
    std::cout << "Graph file: " << params.file_name << std::endl;
    std::cout << "Number of Vertices: " << st.GetNoOfVertices() << std::endl;
    std::cout << "Number of Edges: " << st.GetNoOfEdges() << std::endl;
    std::cout << "Number of Cuts: " << P << std::endl;
    std::cout << "Number of Tasks: " << t_list.size() << std::endl;
    std::cout << "Number of GPUs: " << bbtc.GetNGPU() << std::endl;
    std::cout << "Number of Workers: " << omp_get_max_threads() << std::endl;
    std::cout << "=====================================" << std::endl;

    if( params.border==-1 || params.border>t_list.size() ){
        // all tasks are in gpu
        bbtc.Separator = t_list.size();
    } else if( params.border==-2 ){
        // default setup
        bbtc.Separator = t_list.size()/2;
    } else{
        bbtc.Separator = params.border;
    }

    HooksRegionBegin( "Cuda Initial Memory Allocations and Stream Creation" );
    bbtc.PrepareGpu();
    HooksRegionEnd();

    long n_triangles = 0;
    for( int i=0; i<params.repeat; i++ ) {
        bbtc.ResetFlags();

        if(params.copy_first){
            HooksRegionBegin("Copy tiles to device (s)");
            bbtc.CopyTilesToGpu();
            HooksRegionEnd();
        }

        HooksRegionBegin("Triangle Counting Execution Time (s)");
        n_triangles = bbtc.NoOfTriangles( );
        cudaDeviceSynchronize();
        HooksRegionEnd();
    }

    std::cout << "Number of Triangles: " << n_triangles << std::endl;

    PrintFields();

    st.CleanUp();
    bbtc.CleanUp();
}
