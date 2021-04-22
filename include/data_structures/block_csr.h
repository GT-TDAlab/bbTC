#ifndef BLOCK_CSR_
#define BLOCK_CSR_

#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "definitions.h"

/**
 * @brief: A block csr graph class. This class is simplified for the triangle
 * counting algorithm and doesn't have weight array.
 * Row-major order is used to order tiles in the memory.
 * 
 * @tparam Ordinal: type of the x_adj array elements
 * @tparam Vertex: type of the adj array elements
 **/
template <class Ordinal, class Vertex>
class BlockCSR
{
private:
    Ordinal *cuts_; /**< cut vector of size P_ */
    Ordinal *sparse_tiles_; /**< pointer of the row pointer of the tiles*/
    Ordinal *rows_; /**< row-major ordered block's row pointers */
    Vertex *cols_; /**< row-major ordered block's column indices */
    Ordinal P_; /**< number of cuts */
    Ordinal max_width_; /**< maximum width of the blocks */
    Ordinal max_deg_; /**< maximum degree in the blocks */
    Ordinal max_nnz_; /**< maximum number of nonzeros in a block*/

    float *tile_density_; /**< array to store tile density information */
    float *tile_deg_; /**< array to store tile degree information*/
    float avg_density_; /**< average density of the tiles */
    float tile_deg_limit_=10000; /**< degree limit to define a tile dense*/

public:

    Ordinal NoOfVertices; /**< number of vertices in the graph*/
    Ordinal NoOfEdges; /**< number of edges in the graph*/

    /**
     * @brief: default constructor
     **/
    BlockCSR();

    /**
     * @brief: default destructor
     **/
    void CleanUp();


    /**
     * @brief a constructor that takes a binary block csr graph as input
     * and deserializes it.
     * @param fname absolute path to a binary block csr matrix
     **/
    BlockCSR(std::string fname);

    /**
     * @brief a constructor that takes a csr graph and a partition vector
     * and creates block csr based on this information.
     * 
     * @param cuts a partition vector
     * @param g a pointer for a csr graph
     **/
    BlockCSR(const std::vector<Vertex> &cuts, const std::vector<Ordinal>& x_adj, const std::vector<Vertex>& adj);

    /**
     * @brief returns the maximum width of a tile
     * 
     * @return Ordinal type
     **/
    Ordinal GetMaxWidth();

    /**
     * @brief returns the maximum nonzeros in a tile
     * 
     * @return Ordinal type
     **/
    Ordinal GetMaxNNZ();

    /**
     * @brief returns the nonzeros in the graph
     * 
     * @return Ordinal type
     **/
    Ordinal GetNNZ(Ordinal i);

    /**
     * @brief returns the maximum degree in a tile
     * 
     * @return Ordinal type
     **/
    Ordinal GetMaxDeg();

    /**
     * @brief returns the index of a given row
     * 
     * @return Ordinal type
     **/
    Ordinal GetRow(Ordinal i, Ordinal &src);

    /**
     * @brief returns the size of a tile
     * 
     * @return Ordinal type
     **/
    Ordinal GetSize(Ordinal i);

    /**
     * @brief returns the tile pointer that points to row begins
     * of each tile.
     * 
     * @return Ordinal pointer
     **/
    Ordinal *GetTiles();

    /**
     * @brief returns the row-major ordered row pointer that contains the
     * column pointers for each tile.
     * 
     * @return Ordinal pointer
     **/
    Ordinal *GetRows();

    /**
     * @brief returns the row-major ordered column indices that contains the
     * column pointers for each tile.
     * 
     * @return Vertex pointer
     **/
    Vertex *GetCols();

    /**
     * @brief returns the cut vector.
     * 
     * @return Ordinal pointer
     **/
    Ordinal *GetCuts();

    /**
     * @brief returns the number of tiles.
     * 
     * @return Ordinal
     **/
    Ordinal GetNoOfTiles();

    /**
     * @brief returns the number of cuts.
     * 
     * @return Ordinal
     **/
    Ordinal GetNoOfCuts();

    /**
     * @brief returns the number of vertices
     * 
     * @return Ordinal
     **/
    Ordinal GetNoOfVertices();

    /**
     * @brief returns the number of edges
     * 
     * @return Ordinal
     **/
    Ordinal GetNoOfEdges();

    /**
     * @brief returns the density of the tile[i/P_][i%P_] where i is the
     * row-major order
     * 
     * @return float
     **/
    float GetDensity(Ordinal i);

    /**
     * @brief returns the degree of the tile[i/P_][i%P_] where i is the
     * row-major order
     * 
     * @return float
     **/
    float GetTileDeg(Ordinal i);

    /**
     * @brief returns the width of the tile[i/P_][i%P_] where i is the
     * row-major order
     * 
     * @return Ordinal
     **/
    Ordinal GetWidth(Ordinal i);

    /**
     * @brief returns the height of the tile[i/P_][i%P_] where i is the
     * row-major order
     * 
     * @return Ordinal
     **/
    Ordinal GetHeight(Ordinal i);

    /**
     * @brief sets the dense hashmap size.
     **/
    void SetHashSize();

    /**
     * @brief returns if the tile[i/P_][i%P_] where i is the
     * row-major order is dense or not.
     * 
     * @return bool
     * @note a tile is considered dense if average degree is higher than 10
     **/
    bool IsDense(Ordinal i);

    /**
     * @brief takes a binary block csr formatted file name and deserialize it.
     **/
    void DeSerialize(std::string &file_name);

    /**
     * @brief compute tile attributes such as density, average degree etc.
     **/
    void ComputeTileProps();

    /**
     * @brief takes a csr graph and a partition vector and generates a block
     * csr formatted graph.
     **/
    void CSRtoBCSR(const std::vector<Vertex> &cuts, const std::vector<Ordinal>& x_adj, const std::vector<Vertex>& adj);
};

#include "block_csr.cpp"

#endif // BLOCK_CSR_
