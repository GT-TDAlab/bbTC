#include "bb_tc.h"

void BBTc::InitFlags()
{
    flags_.resize(task_list_.size());

#if defined(ENABLE_CUDA)
    for (int i = 0; i < task_list_.size(); i++)
    {
        cudaMallocHost((void **)&flags_[i], sizeof(std::atomic_flag));
    }

    copy_flags_.resize(n_gpu_, BoolVector(task_list_.size()));
    copier_flags_.resize(n_gpu_, FlagVector(task_list_.size()));
    for (int g = 0; g < n_gpu_; g++)
    {
        for (int i = 0; i < task_list_.size(); i++)
        {
            cudaMallocHost((void **)&copier_flags_[g][i], sizeof(std::atomic_flag));
            copy_flags_[g][i] = false;
        }
    }
#else
    for (int i = 0; i < task_list_.size(); i++)
    {
        flags_[i] = new std::atomic_flag;
    }
#endif
}

void BBTc::ResetFlags()
{
    for (int i = 0; i < task_list_.size(); i++)
    {
        flags_[i]->clear();
    }

#if defined(ENABLE_CUDA)
    for (int g = 0; g < n_gpu_; g++)
    {
        for (int i = 0; i < st_->GetNoOfTiles(); i++)
        {
            copy_flags_[g][i] = false;
            copier_flags_[g][i]->clear();
            cudaFree(rows_in_dev_[g][i]);
            cudaFree(cols_in_dev_[g][i]);
        }
    }
#endif    
}

#if defined(ENABLE_CUDA)
void BBTc::PrepareGpu()
{
    // #pragma omp parallel for schedule( dynamic, 1 )
    for (unsigned int i = 0; i < n_gpu_; i++)
    {
        cudaSetDevice(i);

        cudaMallocHost((void **)&streams_[i], N_STREAMS * sizeof(cudaStream_t));
        for (int j = 0; j < N_STREAMS; j++)
        {
            cudaStreamCreate(&streams_[i][j]);
        }

        cudaMalloc((void **)&nt_dev_[i], N_STREAMS * sizeof(unsigned long long int));
        cudaMemset(nt_dev_[i], 0, N_STREAMS * sizeof(unsigned long long int));

        // bitmap in device
        cudaMalloc((void **)&bitmap_dev_[i], N_STREAMS * sizeof(unsigned int) * BITMAP_BIT_WIDTH * BITMAP_INT_WIDTH);
        cudaMemset(bitmap_dev_[i], 0, N_STREAMS * sizeof(unsigned int) * BITMAP_BIT_WIDTH * BITMAP_INT_WIDTH);
    }
}

void BBTc::CopyTileToGpu(int tile, int gpu_id, int stream_id)
{
    cudaSetDevice(gpu_id);

    if (!copier_flags_[gpu_id][tile]->test_and_set())
    {
        if (copy_flags_[gpu_id][tile])
        {
            copier_flags_[gpu_id][tile]->clear();
            return;
        }

        cudaMalloc(
            (void **)&rows_in_dev_[gpu_id][tile],
            sizeof(Ordinal) * (st_->GetHeight(tile) + 1 + st_->GetNNZ(tile)));

        cudaMemcpyAsync(
            rows_in_dev_[gpu_id][tile],
            &rows_[tiles_[tile]],
            sizeof(Ordinal) * (st_->GetHeight(tile) + 1),
            cudaMemcpyHostToDevice,
            streams_[gpu_id][stream_id]);

        cudaMemcpyAsync(
            &rows_in_dev_[gpu_id][tile][st_->GetHeight(tile) + 1],
            &cols_[rows_[tiles_[tile]]],
            sizeof(Ordinal) * st_->GetNNZ(tile),
            cudaMemcpyHostToDevice,
            streams_[gpu_id][stream_id]);

        cols_in_dev_[gpu_id][tile] = &rows_in_dev_[gpu_id][tile][st_->GetHeight(tile) + 1];
        cudaStreamSynchronize(streams_[gpu_id][stream_id]);
        copy_flags_[gpu_id][tile] = true;
        copier_flags_[gpu_id][tile]->clear();
    }
}

void BBTc::CopyTilesInTaskToGpu(std::vector<size_t> &task, int gpu_id, int stream_id)
{
    for (auto tile : task)
    {
        if (copy_flags_[gpu_id][tile]) continue;
        CopyTileToGpu(tile, gpu_id, stream_id);
    }
}

void BBTc::SyncTaskTileTransfer(std::vector<size_t> &task, int gpu_id, int stream_id)
{
    for (auto tile : task)
    {
        if (copy_flags_[gpu_id][tile]) continue;
        while (copier_flags_[gpu_id][tile]->test_and_set())
        {
        }
        copier_flags_[gpu_id][tile]->clear();
    }
}

void BBTc::SetNGPU(int n)
{
    if (n < n_gpu_)
    {
        n_gpu_ = n;
    }
}

int BBTc::GetNGPU()
{
    return n_gpu_;
}

void BBTc::CopyTilesToGpu( int thread_id ){
  
    int gpu_id = thread_id % n_gpu_; // /N_STREAMS;
    int stream_id = thread_id / n_gpu_; // %N_STREAMS;
    int gpu_task = thread_id;
  
    for( ; gpu_task<task_list_.size(); gpu_task+=n_gpu_worker_ ){
        CopyTilesInTaskToGpu( task_list_[ gpu_task ], gpu_id, stream_id );
    }
}
  
void BBTc::CopyTilesToGpu( ){
  
    n_gpu_worker_ = N_STREAMS*n_gpu_;
  
    #pragma omp parallel for schedule( dynamic )
    for( int i=0; i<n_gpu_worker_; i++ ){
        CopyTilesToGpu( i );
    }
  
    for( int i=0; i<n_gpu_; i++ ){
        cudaSetDevice( i );
        for( int j=0; j<N_STREAMS; j++ ){
            cudaStreamSynchronize( streams_[ i ][ j ] );
        }
    }
}

#endif