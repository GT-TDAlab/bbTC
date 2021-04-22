# bbTC
bbTC: A Block Based Triangle Counting Algorithm for shared memory heterogenous
systems with CPUs and GPUs.

💻 **Source Code:** [http://github.com/GT-TDAlab/bbTC]  

bbTC is developed by the members of [GT-TDAlab](http://tda.gatech.edu). 

## License

bbTC is distributed under BSD License. See [`LICENSE`](LICENSE.md).

## Contributors

- [Abdurrahman Yasar](http://cc.gatech.edu/~ayasar3)
- [Umit V. Catalyurek](http://cc.gatech.edu/~umit)

## Contact

For questions or support contact contributors via their email.

## Citation
Citation for the bbTC (BibTeX):

```bibtex
    @techreport{Yasar20-ArXiv,
        author =  {Abdurrahman Ya\c{s}ar and Sivasankaran Rajamanickam and Jon Berry and {\"{U}}mit V. {\c{C}}ataly{\"{u}}rek},
        title = {A Block Based Triangle Counting Algorithm},
        institution = {ArXiv},
        year = {2020},
        month = {July},
        url = {https://arxiv.org/abs/2009.12457},
        number = {arXiv:2009.12457},
        KEYWORDS = {heterogeneous, block-based, triangle counting},
    }
```

## How to build

Create a `build` subdirectory and run `cmake` and `make` from there like

    cmake ..
    make -j

Note that bbTC targets CPU+GPU execution environments. Hence, if Cuda runtime is
installed in the machine than `nvcc` is used for compilation and outputs an
executable that runs on CPUs and GPUs. Otherwise, `g++` is used for compilation
and outputs an executable that runs on CPUs.

### Dependencies

This project depends on NVIDIA's Cuda Compiler (nvcc) a version greater than
9.0, GCC a version greater than 7.0.0, OpenMP, PIGO I/O library, and SARMA
partitioning library. PIGO and SARMA libraries included under utils directory.
For detailed information:

- [PIGO Library](https://github.com/GT-TDAlab/PIGO)
- [SARMA Library](https://github.com/GT-TDAlab/SARMA)

## How to run

bbTC accepts edge lists, matrix market files, and saved binary CSR files.
bbTC accepts `.bbtc` a binary CSR (Compressed Sparse Row) formatted graph.
Note that if the given graph in text form then after the first execution bbTC
writes binary CSR format of that graph by appending `.bbtc` to the filename.

    bbTC: A Block Based Triangle Counting Algorithm
    syntax: bbtc --graph <path> [OPTION].. [--help]
    Options:                                                         (Defaults:)
    --graph <path>        Path for the graph file.                   ()
                            Supported formats: edge lists, matrix
                            market files and binary CSR.
    --partitioner <str>   Name of the partitioner algorithm.         (pal)
                            Algorithms: [uni | n1d | pal]
                            uni: uniform, checkerboard partitioning
                            n1d: Nicol's 1D Partitioning
                            pal: probe a load partitioning
    --ncut <int>          Number of cuts                             (16)
    --ngpu <int>          Number of GPUs to use                      (all)
                            Note: if 0 then only CPUs used.
    --nstream <int>       Number of streams per GPU                  (4)
                            Note: Cannot be 0.
    --border <int>        Work queue border for GPUs                 (|T|/2)
                            Note: if set to -1, GPUs process all.
    --copy-first          If set then copy graph to GPUs first       (false)
    --repeat <int>        Run bbTC algorithm '--repeat' times        (5)

    Examples:
    bbtc --graph email-EuAll_adj.mtx --partitioner n1d --ncut 4
    bbtc --graph email-EuAll_adj.txt --ncut 4 --copy-first --border 4
    bbtc --graph email-EuAll_adj.el --border 4 --ngpu 2

An example:

    ./bbtc --graph ../email-EuAll.mtx --ncut 8 --partitioner pal
    # Graph read.. N: 265215, NNZ: 364481
    =====================================
    Graph file: ../email-EuAll.mtx
    Number of Vertices: 265216
    Number of Edges: 364481
    Number of Cuts: 8
    Number of Tasks: 120
    Number of Workers: 64
    =====================================
    Number of Triangles: 267313
    Partitioning and Layout construction (s) : 0.0258
    Triangle Counting Execution Time (s) : 0.0030
    Triangle Counting Execution Time (s) : 0.0025
    Triangle Counting Execution Time (s) : 0.0025
    Triangle Counting Execution Time (s) : 0.0025
    Triangle Counting Execution Time (s) : 0.0025

### Primary tested compilers and architectures

- GCC 7.4.0, 8.4.0, 9.2.0, 10.1.0, x86, Ubuntu 18.04
- GCC 7.4.0, powerpc64, Red Hat 7.6
- GCC 10.2.0, x86, macOS Mojave 10.14.6
