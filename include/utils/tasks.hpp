#include <iostream>

#include "csr_graph.h"
#include "block_csr.h"

namespace tasks{

    std::vector<std::vector<size_t>> TaskList (size_t p) {
        std::vector<std::vector<size_t>> t_list;
        for (size_t i=0; i<p; i++) {
            for (size_t j=i; j<p; j++) {
                for (size_t k=j; k<p; k++) {
                    t_list.push_back (std::vector<size_t>{i*p+j, j*p+k, i*p+k});
                }
            }
        }
        return t_list;
    }

    template <typename Ordinal, typename Vertex>
    void SortTasks (std::vector<std::vector<size_t>>&tlist, BlockCSR<Ordinal, Vertex> st) {
        auto weight = [&](std::vector<size_t> t){
            return st.GetNNZ(t[0]) * std::max(st.GetTileDeg(t[1]), st.GetTileDeg(t[2]));
        };

#if defined(ENABLE_CUDA)
        std::sort (tlist.begin(), tlist.end(), [&](const auto &l, const auto &r){
            return weight(l)>weight(r);
        });
#else
        std::sort (tlist.begin(), tlist.end(), [&](const auto &l, const auto &r){
            return weight(l)<weight(r);
        });
#endif  
    }
};
