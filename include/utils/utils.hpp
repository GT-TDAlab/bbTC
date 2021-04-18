#include <iostream>
#include <cmath>
#include <functional>
#include <cstring>
#include <string>
#include <set>

#if defined(ENABLE_CUDA)
#include <cuda.h>
#endif

#include "csr_graph.h"
#include "block_csr.h"
#include "definitions.h"

#include "SARMA/include/sarma.hpp"
#include "PIGO/pigo.hpp"

namespace utils{
    template <typename Ordinal, typename Vertex, typename Weight>
    using PF = std::function<std::vector<Vertex>(CSRGraph<Ordinal,Vertex>&, Vertex)>;

    struct Parameters {
        std::string file_name = "-";
        std::string part_name = "pal";
        int n_gpu = 0;
        int n_stream = N_STREAMS;
        int border = 0;
        int ncut = 32;
        int repeat = 5;
        bool serialize = false;
        bool copy_first = false;
        double alpha = -1;
    };

    inline bool FileExist (const std::string& name) {
        std::ifstream f(name.c_str());
        return f.good();
    }

    std::string tolower(const std::string &s) {
        std::string r;
        for (const auto c: s)
            r.push_back(std::tolower(c));
        return r;
    }

    int strcasecmp(const std::string &a, const std::string &b) {
        return std::strcmp(tolower(a).c_str(), tolower(b).c_str());
    }

    void PrintUsage(){
        std::cout << "bbTC: A Block Based Triangle Counting Algorithm" << std::endl;
        std::cout << "syntax: bbtc --graph <path> [OPTION].. [--help]" << std::endl;
        std::cout << "Options:                                                           (Defaults:)" << std::endl;
        std::cout << "  --graph <path>        Path for the graph file.                   ()" << std::endl;
        std::cout << "                          Supported formats: edge lists, matrix" << std::endl;
        std::cout << "                          market files and binary CSR." << std::endl;
        std::cout << "  --partitioner <str>   Name of the partitioner algorithm.         (pal)" << std::endl;
        std::cout << "                          Algorithms: [uni | n1d | pal]" << std::endl;
        std::cout << "                          uni: uniform, checkerboard partitioning" << std::endl;
        std::cout << "                          n1d: Nicol's 1D Partitioning" << std::endl;
        std::cout << "                          pal: probe a load partitioning" << std::endl;
        std::cout << "  --ncut <int>          Number of cuts                             (16)" << std::endl;
        std::cout << "  --ngpu <int>          Number of GPUs to use                      (all)" << std::endl;
        std::cout << "                          Note: if 0 then only CPUs used." << std::endl;
        std::cout << "  --nstream <int>       Number of streams per GPU                  (4)" << std::endl;
        std::cout << "                          Note: Cannot be 0." << std::endl;
        std::cout << "  --border <int>        Work queue border for GPUs                 (|T|/2)" << std::endl;
        std::cout << "                          Note: if set to -1, GPUs process all." << std::endl;
        std::cout << "  --copy-first          If set then copy graph to GPUs first       (false)" << std::endl;
        std::cout << "  --repeat <int>        Run bbTC algorithm '--repeat' times        (5)" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  bbtc --graph email-EuAll_adj.mtx --partitioner n1d --ncut 4" << std::endl;
        std::cout << "  bbtc --graph email-EuAll_adj.txt --ncut 4 --copy-first --border 4" << std::endl;
        std::cout << "  bbtc --graph email-EuAll_adj.el --border 4 --ngpu 2" << std::endl;
    }

    int ParseArguments(utils::Parameters &params, int argc, const char **argv ){

        if (argc<3){
            PrintUsage();
            std::exit(0);
        }
#if defined(ENABLE_CUDA)
        cudaGetDeviceCount(&params.n_gpu);
#endif
        for (int i = 1; i < argc; i++) {
            if (!strcasecmp(argv[i], "--graph")) {
                params.file_name = std::string(argv[++i]);
            }
            else if (!strcasecmp(argv[i], "--ngpu")) {
                params.n_gpu = std::stoi(argv[++i]);
            }
            else if (!strcasecmp(argv[i], "--nstream")) {
                params.n_stream = N_STREAMS = std::stoi(argv[++i]);
            }
            else if (!strcasecmp(argv[i], "--border")) {
                params.border = std::stoi(argv[++i]);
            }
            else if (!strcasecmp(argv[i], "--repeat")) {
                params.repeat = std::stoi(argv[++i]);
            }
            else if (!strcasecmp(argv[i], "--alpha")) {
                params.alpha = std::stod(argv[++i]);
            }
            else if (!strcasecmp(argv[i], "--serialize")) {
                params.serialize = true;
            }
            else if (!strcasecmp(argv[i], "--copy-first")) {
                params.copy_first = true;
            }
            else if (!strcasecmp(argv[i], "--ncut")) {
                params.ncut = std::stoi(argv[++i]);;
            }
            else if (!strcasecmp(argv[i], "--partitioner")) {
                params.part_name = std::string(argv[++i]);
                if (params.part_name!="pal" &&
                    params.part_name!="n1d" &&
                    params.part_name!="uni"){
                    std::cout << "Wrong partitioning algorithm." << std::endl;
                    PrintUsage();
                    std::exit(0);
                }
            }
            else if (!strcasecmp(argv[i], "--help")) {
                PrintUsage();
                std::exit(0);
            } else {
                std::cout << "Wrong argument '" << argv[i] << "' given." << std::endl;
                PrintUsage();
                std::exit(0);
            }
        }

        if (!FileExist(params.file_name)){
            std::cout << "Graph doesn't exist" << std::endl;
            PrintUsage();
            std::exit(0);
        }

        return 0;
    }

    bool EndsWith (const std::string &str, const std::string &mtch) {
        if(str.size() >= mtch.size() && str.compare(str.size() - mtch.size(), mtch.size(), mtch) == 0)
            return true;
        return false;
    }

    template <typename Ordinal, typename Vertex, typename Weight>
    std::vector<Vertex> UniformCuts (CSRGraph<Ordinal, Vertex> &g, Vertex p){
        std::vector<Vertex> cuts (p+1, 0);

        Vertex nr = g.GetNVertex()/p;

        for( size_t i=1; i<=p; i++ ){
            cuts[ i ] = i*nr;
        }
        cuts[ p ] = g.GetNVertex();

        return cuts;
    }

    template <typename Ordinal, typename Vertex, typename Weight>
    std::vector<Vertex> GreedyCuts (CSRGraph<Ordinal, Vertex> &g, Vertex p){
        std::vector<Vertex> cuts (p+1, 0);
        const auto n = g.GetNVertex();
        const auto x_adj = g.GetVertices();
        Ordinal nr = x_adj[n]/(p)-1;

        for( size_t i=1; i<p; i++ ){
            auto j = std::lower_bound(x_adj, x_adj+n+1, x_adj[cuts[i-1]]+nr) - x_adj;
            cuts[ i ] = j;
        }
        cuts[ p ] = n;
        return cuts;
    }

    template <typename Ordinal, typename Vertex, typename Weight>
    BlockCSR<Ordinal, Vertex> ReadGraph (std::string file_name,  std::string& alg,
            double alpha = -1, const Vertex p=32) {
        ns_filesystem::path graph_path(file_name);
        auto A_ord = std::make_shared<sarma::Matrix<Ordinal, Ordinal>>();

        if (graph_path.extension()==".bbtc" || ns_filesystem::exists(file_name+".bbtc")){
            file_name += (ns_filesystem::exists(file_name+".bbtc")) ? ".bbtc" : "";
            A_ord = std::make_shared<sarma::Matrix<Ordinal, Ordinal>>(file_name, false);
        } else {
            auto A_org = std::make_shared<sarma::Matrix<Ordinal, Ordinal>>(file_name, false);
            auto A_t = std::make_shared<sarma::Matrix<Ordinal, Ordinal>>(A_org->transpose());
            A_org->sort(); A_t->sort();

            auto indptr = std::vector<Ordinal>(A_org->N()+1, 0);
            std::set<Ordinal> testset;
            
            #pragma omp parallel for schedule (dynamic, 1024)
            for (Ordinal i=0; i<A_org->N();  i++){
                Ordinal j=A_org->indptr[i], k=A_t->indptr[i], nnz=0;

                while (j<A_org->indptr[i+1] && k<A_t->indptr[i+1]){
                    if (A_org->indices[j]==A_t->indices[k]){
                        if (A_org->indices[j++]!=i){
                            ++nnz;
                        }
                        ++k;
                    } else if (A_org->indices[j]<A_t->indices[k]){
                        if (A_org->indices[j++]!=i){
                            ++nnz;
                        }
                    } else{
                        if (A_t->indices[k++]!=i){
                            ++nnz;
                        }
                    }
                }

                while (j<A_org->indptr[i+1]){
                    if (A_org->indices[j++]!=i){
                        ++nnz;
                    }
                }

                while (k<A_t->indptr[i+1]){
                    if (A_t->indices[k++]!=i){
                        ++nnz;
                    }
                }
                indptr[i+1] = nnz;
            }

            for (Ordinal i=1; i<=A_org->N();  i++){
                indptr[i] += indptr[i-1];
            }

            auto indices = std::vector<Ordinal>(indptr.back(), 0);
            #pragma omp parallel for schedule (dynamic, 1024)
            for (Ordinal i=0; i<A_org->N();  i++){
                Ordinal j=A_org->indptr[i], k=A_t->indptr[i], l = indptr[i];
                while (j<A_org->indptr[i+1] && k<A_t->indptr[i+1]){
                    if (A_org->indices[j]==A_t->indices[k]){
                        if (A_org->indices[j++]!=i){
                            indices[l++] = A_org->indices[j-1];
                        }
                        ++k;
                    } else if (A_org->indices[j]<A_t->indices[k]){
                        if (A_org->indices[j++]!=i){
                            indices[l++] = A_org->indices[j-1];
                        }
                    } else{
                        if (A_t->indices[k++]!=i){
                            indices[l++] = A_t->indices[k-1];
                        }
                    }
                }

                while (j<A_org->indptr[i+1]){
                    if (A_org->indices[j++]!=i){
                        indices[l++] = A_org->indices[j-1];
                    }
                }

                while (k<A_t->indptr[i+1]){
                    if (A_t->indices[k++]!=i){
                        indices[l++] = A_t->indices[k-1];
                    }
                }
            }

            auto A_sym = std::make_shared<sarma::Matrix<Ordinal, Ordinal>>(
                sarma::Matrix<Ordinal, Ordinal>(indptr, indices, indices, indptr.size())
            );

            A_ord = std::make_shared<sarma::Matrix<Ordinal, Ordinal>>(
                A_sym->order(sarma::Order::ASC, true, true)
            );

            std::cerr << "# Binary graph is saving.." << std::endl;
            auto g_ = pigo::CSR<Ordinal, Ordinal, std::vector<Ordinal>, std::vector<Ordinal>>();
            g_.offsets().assign(A_ord->indptr.begin(), A_ord->indptr.end());
            g_.endpoints().assign(A_ord->indices.begin(), A_ord->indices.end());
            g_.n() = A_ord->indptr.size()-1;
            g_.m() = A_ord->indices.size();
            g_.save(file_name+".bbtc");
        }
        auto A_sp = std::make_shared<sarma::Matrix<Ordinal, Ordinal>>(
            A_ord->sparsify(sarma::utils::get_prob(A_ord->NNZ(), p, p, 10.), 1091467301)
        );

        HooksRegionBegin("Partitioning and Layout construction (s)");

        std::vector<Vertex> cuts;
        if (alg=="uni") {
            cuts = sarma::uniform::partition<Ordinal, Ordinal>(*A_sp, p);
        } else if (alg=="n1d") {
            cuts = sarma::nicol1d::partition<Ordinal, Ordinal>(A_ord->indptr, p);
        } else {
            A_sp->get_sps();
            cuts = sarma::probe_a_load::partition<Ordinal, Ordinal>(*A_sp, p);
        }
        cuts[p] = A_ord->N()+1;


        CSRGraph<Ordinal, Vertex> g(A_ord->indptr.data(), A_ord->indices.data(), A_ord->N()+1, A_ord->indices.size());

        auto bcsr = BlockCSR<Ordinal, Vertex> (cuts, &g);

        HooksRegionEnd();

        return bcsr;
    }

    template <typename Ordinal, typename Vertex, typename Weight>
    BlockCSR<Ordinal, Vertex> BlockGraph (const std::string file_name, std::string& alg,
            double alpha=-1, const Vertex p=32,
            utils::PF<Ordinal, Vertex, Weight> *fp=nullptr) {
        if (utils::EndsWith (file_name, ".bcsr")) {
            return BlockCSR<Ordinal, Vertex> (file_name);
        } else{
            return ReadGraph<Ordinal, Vertex, Weight> (file_name, alg, alpha, p);
        }
    }
};
