#ifndef CSR_TC_PAR_GPU_S_
#define CSR_TC_PAR_GPU_S_

#include <iostream>
#include <atomic>
#include <omp.h>

#include "block_csr.h"
#include "hooks.hpp"

#if defined(ENABLE_CUDA)
#include <cuda.h>
#include "bb_tc_gpu.h"
#endif

using FlagVector = std::vector<std::atomic_flag *>;
using FlagVector2D = std::vector<FlagVector>;

using OVector = std::vector<Ordinal *>;
using OVector2D = std::vector<OVector>;

using BoolVector = std::vector<bool>;
using BoolVector2D = std::vector<BoolVector>;

/**
 * @brief main class for the bbTC algorithm
 **/
class BBTc
{

private:
    //graph variables
    BlockCSR<Ordinal, Vertex> *st_; /**< a pointer to block csr */
    Ordinal *tiles_; /**< a pointer that point beginning of rows in blocks*/
    Ordinal *rows_; /**< a pointer to row pointers of tiles */
    Ordinal P_; /**< number of cuts*/
    Vertex *cols_; /**< a pointer to the column indices of blocks*/
    std::vector<std::vector<size_t>> task_list_; /**< task list */
    std::vector<Ordinal *> hmaps_; /**< hmap pool for workers */
    std::atomic_long no_of_triangles_; /**< global number of triangles*/
    unsigned long long int nt_host_; /**< number of triangles in host*/
    std::atomic_int cpu_task_; /**< last processed cpu_task */
    int n_worker_; /**< number of worker threads */

    FlagVector flags_;

#if defined(ENABLE_CUDA)
    std::vector<unsigned int *> bitmap_dev_; /**< bitmap hashmaps for devices*/
    std::vector<unsigned long long int *> nt_dev_; /**< number of devices */
    std::vector<cudaStream_t *> streams_; /**< vector of stream threads */
    int n_gpu_worker_; /**< number of gpu worker threads*/
    int n_gpu_; /**< number of gpus*/
    OVector2D rows_in_dev_; /**< pointers to row pointers in the devices*/
    OVector2D cols_in_dev_; /**< pointers to column indices in the devices*/
    FlagVector2D copier_flags_; /**< assignee flag that a thread sets whe copy start*/
    BoolVector2D copy_flags_; /**< flag that is set when the copy is done */
    std::atomic_int gpu_task_; /**< number of triangles in devices */
#endif

public:
    int Separator; /**< border cut-off between gpus and cpus*/

    /**
     * @brief constructor to initiate bbTC class
     * 
     * @param st a pointer to block csr
     * @param tasks task list
     * @param P number of cuts
     **/
    BBTc(BlockCSR<Ordinal, Vertex> *st, std::vector<std::vector<size_t>> &tasks, Ordinal P);

    /**
     * @brief clean up method
     **/
    void CleanUp();

    /**
     * @brief alloacate and initialize hashmaps both on device and host
     **/
    void ArrangeHashMaps();

    /**
     * @brief executer method that starts and runs workers gathers results
     * and outputs the result.
     * 
     * @return number of triangles
     **/
    unsigned long long int NoOfTriangles();

    /**
     * @brief starts cpu workers and assignes new tasks until there exist
     **/
    void RunCpuKernels();

    /**
     * @brief counts number of triangles in task i using CPUs
    **/
    void NoOfTrianglesInCpus(int i);

    /**
     * @brief counts triangles in a sparse task on CPUs
     * 
     * @param ab beginning pointer for the first tile: (u, v)
     * @param bc beginning pointer for the second tile: (v, w)
     * @param ac beginning pointer for the third tile: (u, w)
     **/
    void TrianglesInASparseTask(Ordinal ab, Ordinal bc, Ordinal ac);

    /**
     * @brief counts triangles in a dense task on CPUs
     * 
     * @param ab beginning pointer for the first tile: (u, v)
     * @param bc beginning pointer for the second tile: (v, w)
     * @param ac beginning pointer for the third tile: (u, w)
     **/
    void TrianglesInADenseTask(Ordinal ab, Ordinal bc, Ordinal ac);

    /**
     * @brief counts common neighbors using hashmap-based list intersection
     * 
     * @param ab beginning pointer for the first tile: (u, v)
     * @param bc beginning pointer for the second tile: (v, w)
     * @param ac beginning pointer for the third tile: (u, w)
     * @param f_i id of a vertex: u
     * @param worker_id CPU worker thread id
     **/
    long ListIntersectUsingHMap(Ordinal ab_src, Ordinal bc_src, Ordinal ac_src, Ordinal f_i, int worker_id);

    /**
     * @brief counts common neighbors using merge based list intersection
     * 
     * @param ab beginning pointer for the first tile: (u, v)
     * @param bc beginning pointer for the second tile: (v, w)
     * @param ac beginning pointer for the third tile: (u, w)
     * @param f_i id of a vertex: u
     **/
    long ListIntersect(Ordinal ab_src, Ordinal bc_src, Ordinal ac_src, Ordinal f_i);

    /**
     * @brief initialize task assignement flags
     **/
    void InitFlags();

    /**
     * @brief reset task assignement flags
     **/
    void ResetFlags();

#if defined(ENABLE_CUDA)
    /**
     * @brief initialize, allocate GPU specific datastructures
     **/
    void PrepareGpu();

    /**
     * @brief starts GPU workers and assignes new tasks until there exist
     * @param thread_id streaming thread id
     **/
    void RunGpuKernels(int thread_id);

    /**
     * @brief count triangles using gpu
     * 
     * @param i task id
     * @param gpu_id which gpu
     * @param stream_id which stream on gpu
     **/
    void NoOfTrianglesInGPU(int i, int gpu_id, int stream_id);

    /**
     * @brief copy next tiles for that gpu of thread_id
     * @param thread_id worker thread id of a gpu
     **/
    void CopyTilesToGpu( int thread_id );

    /**
     * @brief copy next tiles for that gpu of thread_id
     **/
    void CopyTilesToGpu( );

    /**
     * @brief copy tile for a gpu and stream
     * @param tile tile id
     * @param gpu_id gpu id
     * @param stream_id which strean on that gpu
     **/
    void CopyTileToGpu(int tile, int gpu_id, int stream_id);

    /**
     * @brief copy tiles in a task for that gpu of thread_id
     * @param task list of tasks
     * @param gpu_id gpu id
     * @param stream_id which strean on that gpu
     **/
    void CopyTilesInTaskToGpu(std::vector<size_t> &task, int gpu_id, int stream_id);

    /**
     * @brief synchronize cuda stream for processing next task
     * @param task list of tasks
     * @param gpu_id gpu id
     * @param stream_id which strean on that gpu
     **/
    void SyncTaskTileTransfer(std::vector<size_t> &task, int gpu_id, int stream_id);

    /**
     * @brief set number of gpus that are going to be used
     * @param n number of gpus
     **/
    void SetNGPU(int n);

    /**
     * @brief get number of gpus that are going to be used
     * 
     * @return number of gpus considered
     **/
    int GetNGPU();
#endif
};

#include "triangle_counting/bb_tc.cpp"
#include "triangle_counting/bb_tc_helper.cpp"
#include "triangle_counting/cpu_dense_kernel.cpp"
#include "triangle_counting/cpu_sparse_kernel.cpp"

#endif
