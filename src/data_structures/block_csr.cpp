#include "block_csr.h"

template <class Ordinal, class Vertex>
BlockCSR<Ordinal, Vertex>::BlockCSR(){};

template <class Ordinal, class Vertex>
void BlockCSR<Ordinal, Vertex>::CleanUp(){
#if defined(ENABLE_CUDA)
    cudaFree(cuts_);
    cudaFree(sparse_tiles_);
    cudaFree(rows_);
    cudaFree(cols_);
    cudaFree(tile_density_);
    cudaFree(tile_deg_);
#else
    delete[] cuts_;
    delete[] sparse_tiles_;
    delete[] rows_;
    delete[] cols_;
    delete[] tile_density_;
    delete[] tile_deg_;
#endif
}

template <class Ordinal, class Vertex>
BlockCSR<Ordinal, Vertex>::BlockCSR(std::string fname)
{
    max_nnz_ = 0;
    max_width_ = 0;

    DeSerialize(fname);
}

template <class Ordinal, class Vertex>
BlockCSR<Ordinal, Vertex>::BlockCSR(const std::vector<Vertex> &cuts, CSRGraph<Ordinal, Vertex> *g)
{
    P_ = cuts.size() - 1;
    NoOfVertices = g->GetNVertex();
    NoOfEdges = g->GetNEdge();
    CSRtoBCSR(cuts, g);
    ComputeTileProps();
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetMaxWidth()
{
    if (max_width_ > 0)
    {
        return max_width_;
    }

    Ordinal temp = 0;
    for (Ordinal i = 1; i <= P_; i++)
    {
        temp = cuts_[i] - cuts_[i - 1];
        if (temp > max_width_)
        {
            max_width_ = temp;
        }
    }

    return max_width_;
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetMaxNNZ()
{
    if (max_nnz_ > 0)
    {
        return max_nnz_;
    }

    max_nnz_ = 0;
    Ordinal temp = 0;
    for (Ordinal i = 0; i < GetNoOfTiles(); i++)
    {
        temp = rows_[sparse_tiles_[i + 1]] - rows_[sparse_tiles_[i]];
        if (temp > max_nnz_)
        {
            max_nnz_ = temp;
        }
    }

    return max_nnz_;
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetNNZ(Ordinal i)
{
    return rows_[sparse_tiles_[i + 1]] - rows_[sparse_tiles_[i]];
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetMaxDeg()
{
    return max_deg_;
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetRow(Ordinal i, Ordinal &src)
{

    Ordinal len = sparse_tiles_[i + 1] - sparse_tiles_[i];
    src = sparse_tiles_[i];

    return len;
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetSize(Ordinal i)
{
    return GetNNZ(i) + GetHeight(i);
}

template <class Ordinal, class Vertex>
Ordinal *BlockCSR<Ordinal, Vertex>::GetTiles()
{
    return sparse_tiles_;
}

template <class Ordinal, class Vertex>
Ordinal *BlockCSR<Ordinal, Vertex>::GetRows()
{
    return rows_;
}

template <class Ordinal, class Vertex>
Vertex *BlockCSR<Ordinal, Vertex>::GetCols()
{
    return cols_;
}

template <class Ordinal, class Vertex>
Ordinal *BlockCSR<Ordinal, Vertex>::GetCuts()
{
    return cuts_;
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetNoOfTiles()
{
    return P_ * P_;
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetNoOfCuts()
{
    return P_;
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetNoOfVertices()
{
    return NoOfVertices;
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetNoOfEdges()
{
    return NoOfEdges;
}

template <class Ordinal, class Vertex>
float BlockCSR<Ordinal, Vertex>::GetDensity(Ordinal i)
{
    return tile_density_[i];
}

template <class Ordinal, class Vertex>
float BlockCSR<Ordinal, Vertex>::GetTileDeg(Ordinal i)
{
    return tile_deg_[i];
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetWidth(Ordinal i)
{
    Ordinal t = i % P_;
    return cuts_[t + 1] - cuts_[t];
}

template <class Ordinal, class Vertex>
Ordinal BlockCSR<Ordinal, Vertex>::GetHeight(Ordinal i)
{
    return sparse_tiles_[i + 1] - sparse_tiles_[i];
}

template <class Ordinal, class Vertex>
void BlockCSR<Ordinal, Vertex>::SetHashSize()
{
    // tile_deg_limit_ = 10; //tiles_degs[ limit_index ];
    HASH_LIMIT = 0;

    for (Ordinal i = 0; i < GetNoOfTiles(); i++)
    {
        // std::cout << GetTileDeg( i ) << " ";
        if (GetTileDeg(i) > tile_deg_limit_)
        {
            if (GetWidth(i) > HASH_LIMIT)
            {
                HASH_LIMIT = GetWidth(i);
            }
        }
    }
    // std::cout << std::endl;
}

template <class Ordinal, class Vertex>
bool BlockCSR<Ordinal, Vertex>::IsDense(Ordinal i)
{
    return GetTileDeg(i) > tile_deg_limit_;
}

/* SERIALISER */
template <class Ordinal, class Vertex>
void BlockCSR<Ordinal, Vertex>::DeSerialize(std::string &file_name)
{
    std::ifstream graph_file_;
    graph_file_.open(file_name, std::ios::binary | std::ios::in);

    graph_file_.read((char *)&NoOfVertices, sizeof(Ordinal));
    graph_file_.read((char *)&NoOfEdges, sizeof(Ordinal));
    graph_file_.read((char *)&P_, sizeof(Ordinal));

    Ordinal no_rows = NoOfVertices;
    Ordinal non_zeros = NoOfEdges;
    Ordinal no_cuts = P_ + 1;
    Ordinal no_tiles = P_ * P_ + 1;
    Ordinal no_rpart = no_rows * P_;

#if defined(ENABLE_CUDA)
    cudaMallocHost((void **)&cuts_, no_cuts * sizeof(Ordinal));
    cudaMallocHost((void **)&sparse_tiles_, no_tiles * sizeof(Ordinal));
    cudaMallocHost((void **)&rows_, no_rpart * sizeof(Ordinal));
    cudaMallocHost((void **)&cols_, non_zeros * sizeof(Vertex));
#else
    cuts_ = new Ordinal[ no_cuts ];
    sparse_tiles_ = new Ordinal[ no_tiles ];
    rows_ = new Ordinal[ no_rpart ];
    cols_ = new Vertex[ non_zeros ];
#endif

    graph_file_.read((char *)cuts_, (no_cuts) * sizeof(Ordinal));
    graph_file_.read((char *)sparse_tiles_, (no_tiles) * sizeof(Ordinal));
    graph_file_.read((char *)rows_, (no_rpart) * sizeof(Ordinal));
    graph_file_.read((char *)cols_, (non_zeros) * sizeof(Vertex));

    // filling helper data
    tile_density_ = new float[no_tiles];
    tile_deg_ = new float[no_tiles];

    ComputeTileProps();

    graph_file_.close();
}

template <class Ordinal, class Vertex>
void BlockCSR<Ordinal, Vertex>::ComputeTileProps()
{
    avg_density_ = NoOfEdges / ((P_) * (P_ + 1) / 2);

    #pragma omp parallel for schedule( dynamic, 32 )
    for (Ordinal i = 0; i < GetNoOfTiles(); i++)
    {
        tile_density_[i] = 0;
        tile_deg_[i] = 0;
    }

    #pragma omp parallel for schedule( dynamic, 2048 )
    for (Ordinal i = 0; i < GetNoOfTiles(); i++)
    {
        Ordinal max_deg = 0;
        for (Ordinal j = sparse_tiles_[i]; j < sparse_tiles_[i + 1]; j++)
        {
            max_deg_ = std::max(max_deg_, rows_[j + 1] - rows_[j]);
            max_deg += rows_[j + 1] - rows_[j];
        }
        tile_deg_[i] = (float)max_deg / (sparse_tiles_[i + 1] - sparse_tiles_[i]);
        tile_density_[i] = (1.0 / GetWidth(i)) * (1.0 / GetHeight(i));
        tile_density_[i] = tile_density_[i] * GetSize(i);
    }

    SetHashSize();
}

// convert csr graph to block csr
template <class Ordinal, class Vertex>
void BlockCSR<Ordinal, Vertex>::CSRtoBCSR(const std::vector<Vertex> &cuts, CSRGraph<Ordinal, Vertex> *g)
{
    const auto x_adj = g->GetVertices();
    const auto adj = g->GetEdges();

    P_ = cuts.size() - 1;
    NoOfVertices = g->GetNVertex();;
    NoOfEdges = g->GetNEdge();

    size_t n_tiles = P_ * P_ + 1;
    size_t n_rpart = (NoOfVertices + 1) * P_;
    
    // memory allocation for the block csr
#if defined(ENABLE_CUDA)
    cudaMallocHost((void **)&sparse_tiles_, n_tiles * sizeof(Ordinal));
    cudaMallocHost((void **)&rows_, n_rpart * sizeof(Ordinal));
    cudaMallocHost((void **)&cols_, NoOfEdges * sizeof(Ordinal));
    cudaMallocHost((void **)&cuts_, (P_ + 1) * sizeof(Vertex));
#else
    cuts_ = new Ordinal[ P_+1 ];
    sparse_tiles_ = new Ordinal[ n_tiles ];
    rows_ = new Ordinal[ n_rpart ];
    cols_ = new Vertex[ NoOfEdges ];
#endif
    for (size_t i = 0; i <= P_; i++)
    {
        cuts_[i] = cuts[i];
    }

    // initialization sparse tile to point rows
    sparse_tiles_[0] = 0;
    for (size_t i = 0; i < P_; i++)
    {
        for (size_t j = 0; j < P_; j++)
        {
            sparse_tiles_[i * P_ + j + 1] = sparse_tiles_[i * P_ + j];
            sparse_tiles_[i * P_ + j + 1] += cuts[i + 1] - cuts[i];
        }
    }

    // initializing row pointers
    #pragma omp parallel for schedule( dynamic, 2048 ) 
    for (size_t i = 0; i < n_rpart; i++)
    {
        rows_[i] = 0;
    }

    #pragma omp parallel for schedule( dynamic, 2048 )
    for (size_t u = 0; u < NoOfVertices; u++)
    {
        size_t x = 0, y = 0;
        while (u >= cuts[y + 1])
        {
            ++y;
        }

        for (size_t v = x_adj[u]; v < x_adj[u + 1]; v++)
        {
            while (adj[v] >= cuts[x + 1])
            {
                ++x;
            }
            rows_[sparse_tiles_[y * P_ + x] + (u - cuts[y]) + 1] += 1;
        }
    }

    for (size_t i = 1; i < n_rpart; i++)
    {
        rows_[i] += rows_[i - 1];
    }


    #pragma omp parallel for schedule( dynamic, 2048 )
    for (size_t u = 0; u < NoOfVertices; u++)
    {
        size_t x = 0, y = 0;
        while (u >= cuts[y + 1])
        {
            ++y;
        }

        size_t i = rows_[sparse_tiles_[y * P_] + (u - cuts[y])];;
        for (size_t v = x_adj[u]; v < x_adj[u + 1]; v++)
        {
            while (adj[v]>=cuts[x+1])
            {
                i = rows_[sparse_tiles_[y * P_ + (++x)] + (u - cuts[y])];
            }
            cols_[i++] = adj[v] - cuts_[x];
        }
    }

    tile_density_ = new float[n_tiles];
    tile_deg_ = new float[n_tiles];
}
