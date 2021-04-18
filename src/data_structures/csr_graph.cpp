#include "csr_graph.h"

template <class Ordinal, class Vertex>
CSRGraph<Ordinal, Vertex>::CSRGraph()
{

}

template <class Ordinal, class Vertex>
CSRGraph<Ordinal, Vertex>::CSRGraph(const Ordinal *A, const Vertex *B, size_t n, size_t m)
{
    x_adj_ = new Ordinal[n+1];
    adj_ = new Vertex[m];
    n_vertex_ = n;
    n_edges_ = m;

    for (size_t i=0; i<=n; i++){
        x_adj_[i] = A[i];
    }

    for (size_t i=0; i<m; i++){
        adj_[i] = B[i];
    }
}

template <class Ordinal, class Vertex>
CSRGraph<Ordinal, Vertex>::~CSRGraph()
{
    delete[] x_adj_;
    delete[] adj_;
    vertex_degrees_.clear();
    vertex_order_.clear();
}

template <class Ordinal, class Vertex>
size_t CSRGraph<Ordinal, Vertex>::GetNVertex()
{
    return n_vertex_;
}

template <class Ordinal, class Vertex>
size_t CSRGraph<Ordinal, Vertex>::GetNEdge()
{
    return n_edges_;
}

template <class Ordinal, class Vertex>
Ordinal *CSRGraph<Ordinal, Vertex>::GetVertices()
{
    return x_adj_;
}

template <class Ordinal, class Vertex>
Vertex *CSRGraph<Ordinal, Vertex>::GetEdges()
{
    return adj_;
}

template <class Ordinal, class Vertex>
void CSRGraph<Ordinal, Vertex>::SortVertices(Ordinal *A, Vertex *B, size_t n, size_t m)
{
    vertex_degrees_.resize(n);

#pragma omp parallel for schedule(dynamic)
    for (size_t u = 0; u < n; u++)
    {
        vertex_degrees_[u] = std::make_pair(A[u + 1] - A[u], u);
    }

    std::sort(vertex_degrees_.begin(), vertex_degrees_.end(), [](auto &lhs, auto &rhs) {
        if (lhs.first == rhs.first)
        {
            return lhs.second < rhs.second;
        }

        return lhs.first < rhs.first;
    });
}

template <class Ordinal, class Vertex>
Vertex CSRGraph<Ordinal, Vertex>::OrderVertices(Ordinal *A, Vertex *B, size_t n, size_t m)
{
    Vertex maxval = n + 1;
    const auto &beg = std::lower_bound(vertex_degrees_.begin(), vertex_degrees_.end(), 2, [](auto &lhs, auto &rhs) {
                          return lhs.first < rhs;
                      }) -
                      vertex_degrees_.begin();

#pragma omp parallel for schedule(dynamic)
    for (size_t u = 0; u < n; u++)
    {
        auto order = (u < beg) ? maxval : u - beg;
        vertex_order_[vertex_degrees_[u].second] = order;
    }

    return beg;
}

template <class Ordinal, class Vertex>
void CSRGraph<Ordinal, Vertex>::SortAndOrder(Ordinal *A, Vertex *B, size_t n, size_t m, bool filter)
{
    Vertex maxval = n + 1;

    vertex_order_.resize(n);
    SortVertices(A, B, n, m);
    auto beg = OrderVertices(A, B, n, m);

    if (filter)
    {
        beg = 0;
    }

    n_vertex_ = n - beg;
    x_adj_ = new Ordinal[n_vertex_ + 1];
    x_adj_[0] = 0;

#pragma omp parallel for schedule(dynamic)
    for (size_t i = beg; i < n; i++)
    {
        const auto u = vertex_degrees_[i].second;
        Ordinal deg = 0;

        for (size_t v = A[u]; v < A[u + 1]; v++)
        {
            if (vertex_order_[B[v]] == maxval || vertex_order_[u] > vertex_order_[B[v]])
            {
                continue;
            }
            ++deg;
        }
        if (filter)
        {
            this->x_adj_[u + 1] = deg;
        }
        else
        {
            this->x_adj_[vertex_order_[u] + 1] = deg;
        }
    }

    // ParallelScan(n_vertex_ + 1, temp, x_adj_);
    for (size_t i = 0; i < n_vertex_; i++)
    {
        this->x_adj_[i + 1] += this->x_adj_[i];
    }

    n_edges_ = x_adj_[n_vertex_];
    adj_ = new Vertex[n_edges_];

#pragma omp parallel for schedule(dynamic)
    for (size_t i = beg; i < n; i++)
    {
        const auto u = vertex_degrees_[i].second;
        const auto st = x_adj_[((filter) ? u : vertex_order_[u])];
        Ordinal deg = 0;

        for (size_t v = A[u]; v < A[u + 1]; v++)
        {
            if (vertex_order_[B[v]] == maxval || vertex_order_[u] > vertex_order_[B[v]])
            {
                continue;
            }

            if (filter)
            {
                adj_[st + deg++] = B[v];
            }
            else
            {
                adj_[st + deg++] = vertex_order_[B[v]];
            }
        }

        std::sort(adj_ + st, adj_ + st + deg);
    }
}