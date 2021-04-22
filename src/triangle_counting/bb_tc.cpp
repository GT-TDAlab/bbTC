#include "bb_tc.h"

BBTc::BBTc(BlockCSR<Ordinal, Vertex> *st, std::vector<std::vector<size_t>> &tasks, Ordinal P)
{
    tiles_ = st->GetTiles();
    rows_ = st->GetRows();
    cols_ = st->GetCols();

    // tasks_ = tasks;
    task_list_ = tasks;
    st_ = st;
    P_ = P;
    Separator = -1;

    ArrangeHashMaps();

#if defined(ENABLE_CUDA)
    // Memory allocations for GPU
    cudaGetDeviceCount(&n_gpu_);

    // Synchronize all gpus
    for (unsigned int i = 0; i < n_gpu_; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    nt_dev_.resize(n_gpu_);
    bitmap_dev_.resize(n_gpu_);
    streams_.resize(n_gpu_);
    rows_in_dev_.resize(n_gpu_, OVector(st_->GetNoOfTiles(), NULL));
    cols_in_dev_.resize(n_gpu_, OVector(st_->GetNoOfTiles(), NULL));
#endif
    nt_host_ = 0;
}

void BBTc::CleanUp()
{
    for (auto hmap : hmaps_){
        delete[] hmap;
    }

    task_list_.clear();
#if defined(ENABLE_CUDA)
    for (auto b : bitmap_dev_){
        cudaFree(b);
    }

    for (auto b : nt_dev_){
        cudaFree(b);
    }

    for (auto b : streams_){
        cudaFree(b);
    }

    for (auto b : rows_in_dev_){
        for (auto i : b){
            cudaFree(i);
        }
    }

    for (auto b : cols_in_dev_){
        for (auto i : b){
            cudaFree(i);
        }
    }

    copier_flags_.clear();
    copy_flags_.clear();
#endif

}

#if defined(ENABLE_CUDA)
void BBTc::NoOfTrianglesInGPU(int i, int gpu_id, int stream_id)
{
    Ordinal ab, bc, ac;

    int blockSize = 256, numBlocks = 1;

    ab = task_list_[i][0];
    bc = task_list_[i][1];
    ac = task_list_[i][2];

    if (st_->GetTileDeg(bc) == 0 ||
        st_->GetTileDeg(ac) == 0 ||
        st_->GetTileDeg(ab) == 0)
    {
        return;
    }

    cudaSetDevice(gpu_id);
    SyncTaskTileTransfer(task_list_[i], gpu_id, stream_id);

    numBlocks = (st_->GetHeight(ab) + blockSize - 1) / blockSize;

   if (st_->GetWidth(ac) < BITMAP_BIT_WIDTH && st_->GetHeight(ab) < BITMAP_BIT_WIDTH)
   {
       DenseKernel<<<numBlocks, blockSize, 0, streams_[gpu_id][stream_id]>>>(
           rows_in_dev_[gpu_id][ab], cols_in_dev_[gpu_id][ab], st_->GetHeight(ab),
           rows_in_dev_[gpu_id][bc], cols_in_dev_[gpu_id][bc],
           rows_in_dev_[gpu_id][ac], cols_in_dev_[gpu_id][ac],
           &nt_dev_[gpu_id][stream_id],
           &bitmap_dev_[gpu_id][(stream_id)*BITMAP_INT_WIDTH * BITMAP_BIT_WIDTH]);
   }
   else
   {
        SparseKernel<<<numBlocks, blockSize, 0, streams_[gpu_id][stream_id]>>>(
            rows_in_dev_[gpu_id][ab], cols_in_dev_[gpu_id][ab], st_->GetHeight(ab),
            rows_in_dev_[gpu_id][bc], cols_in_dev_[gpu_id][bc],
            rows_in_dev_[gpu_id][ac], cols_in_dev_[gpu_id][ac],
            &nt_dev_[gpu_id][stream_id]);
   }
}
#endif

void BBTc::NoOfTrianglesInCpus(int i)
{
    Ordinal ab = task_list_[i][0];
    Ordinal bc = task_list_[i][1];
    Ordinal ac = task_list_[i][2];

    if (st_->GetTileDeg(bc) == 0 ||
        st_->GetTileDeg(ac) == 0 ||
        st_->GetTileDeg(ab) == 0)
    {
        return;
    }

    if (st_->IsDense(ac) || st_->IsDense(bc) || st_->IsDense(ab))
    {
        TrianglesInADenseTask(ab, bc, ac);
    }
    else
    {
        TrianglesInASparseTask(ab, bc, ac);
    }
}

#if defined(ENABLE_CUDA)
void BBTc::RunGpuKernels(int thread_id)
{
    int gpu_id = thread_id % n_gpu_;    // /N_STREAMS;
    int stream_id = thread_id / n_gpu_; // %N_STREAMS;
    int gpu_task = thread_id;

    CopyTilesInTaskToGpu(task_list_[gpu_task], gpu_id, stream_id);
    for (; gpu_task < Separator; gpu_task += n_gpu_worker_)
    {
        NoOfTrianglesInGPU(gpu_task, gpu_id, stream_id);
        if (gpu_task + n_gpu_worker_ < Separator)
        {
            CopyTilesInTaskToGpu(task_list_[gpu_task + n_gpu_worker_], gpu_id, stream_id);
        }
    }

    if (gpu_task < task_list_.size())
    {
        CopyTilesInTaskToGpu(task_list_[gpu_task], gpu_id, stream_id);
    }

    while (gpu_task < task_list_.size())
    {
        if (!flags_[gpu_task]->test_and_set())
        {
            NoOfTrianglesInGPU(gpu_task, gpu_id, stream_id);
        }
        else
        {
            break;
        }

        gpu_task += n_gpu_worker_;

        if (gpu_task < task_list_.size())
        {
            CopyTilesInTaskToGpu(task_list_[gpu_task], gpu_id, stream_id);
        }
    }
}
#endif

void BBTc::RunCpuKernels()
{

    int cpu_task = cpu_task_.fetch_add(-1);

    while (cpu_task >= Separator)
    {
        if (!flags_[cpu_task]->test_and_set())
        {
            NoOfTrianglesInCpus(cpu_task);
        }
        cpu_task = cpu_task_.fetch_add(-1);
    }
}

#if defined(ENABLE_CUDA)
unsigned long long int BBTc::NoOfTriangles()
{
    // if there are less tasks than workers
    if (task_list_.size()<n_worker_) {
        n_worker_ = task_list_.size();
    }

    // if there are too many streams adjust it
    if (N_STREAMS*n_gpu_ > n_worker_) {
        N_STREAMS = n_worker_/n_gpu_;
    }

    no_of_triangles_ = 0;
    for (int i = 0; i < n_gpu_; i++)
    {
        cudaSetDevice(i);
        cudaMemset(nt_dev_[i], 0, N_STREAMS * sizeof(unsigned long long int));
    }
    n_gpu_worker_ = N_STREAMS * n_gpu_;

    cpu_task_ = task_list_.size() - 1;
    gpu_task_ = Separator;

    #pragma omp parallel for schedule(static)
    for (int i = n_worker_ - 1; i >= 0; i--)
    {
        if (i < n_gpu_worker_)
        {
            RunGpuKernels(i);
        }
        else
        {
            RunCpuKernels();
        }
    }

    for (int i = 0; i < n_gpu_; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < n_gpu_; i++)
    {
        for (int j = 0; j < N_STREAMS; j++)
        {
            cudaMemcpy( &nt_host_, &nt_dev_[i][j], sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
            no_of_triangles_.fetch_add(nt_host_);
        }
    }

    return no_of_triangles_;
}
#else

unsigned long long int BBTc::NoOfTriangles()
{
    // if there are less tasks than workers
    if (task_list_.size()<n_worker_) {
        n_worker_ = task_list_.size();
    }

    no_of_triangles_ = 0;

    cpu_task_ = task_list_.size() - 1;

    #pragma omp parallel for schedule(static)
    for (int i = 0 ; i < n_worker_; i++)
    {
        RunCpuKernels();
    }

    return no_of_triangles_;
}

#endif
