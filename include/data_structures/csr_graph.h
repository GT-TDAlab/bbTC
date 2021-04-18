#ifndef CSRGRAPH_HPP_
#define CSRGRAPH_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>

/**
 * @brief: A simple csr graph class. This class is simplified for the triangle
 * counting algorithm and doesn't have weight array.
 * 
 * @tparam Ordinal: type of the x_adj array elements
 * @tparam Vertex: type of the adj array elements
 **/
template <class Ordinal, class Vertex>
class CSRGraph
{

private:
    Ordinal *x_adj_; /**< x_adj array */
    Vertex *adj_; /**< adj array */
    size_t n_vertex_, n_edges_; /**< number of vertices and edges */

    std::vector<std::pair<size_t, Vertex>> vertex_degrees_;/**< used to stores vertex degrees */
    std::vector<Vertex> vertex_order_; /**< vertex order map after sorting */

public:

    CSRGraph();

    CSRGraph(const Ordinal *A, const Vertex *B, size_t n, size_t m);

    /**
     * @brief a destructor
     **/
    ~CSRGraph();

    /**
     * @brief used to order vertices after sorting the vertices based on their
     * degrees
     **/
    Vertex OrderVertices(Ordinal *A, Vertex *B, size_t n, size_t m);

    /**
     * @brief Allocates vertex_degrees_, initializes and sorts
     **/
    void SortVertices(Ordinal *A, Vertex *B, size_t n, size_t m);

    /**
     * @brief Primary method that is called to order vertices basd on their
     * dgrees and creating an ordered graph.
     **/
    void SortAndOrder(Ordinal *A, Vertex *B, size_t n, size_t m, bool filter = false);

    /**
     * @brief returns number of vertices in the graphs
     * @return size_t type
     **/
    size_t GetNVertex();

    /**
     * @brief returns number of edges in the graphs
     * @return size_t type
     **/
    size_t GetNEdge();

    /**
     * @brief returns x_adj array
     * @return an Ordinal pointer of size n_vertex_
     **/
    Ordinal *GetVertices();

    /**
     * @brief returns adj array
     * @return a Vertex pointer of size n_edge_
     **/
    Vertex *GetEdges();    
};

#include "csr_graph.cpp"

#endif // CSR_SPARSE_TILES_H