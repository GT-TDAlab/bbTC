#pragma once

#include <experimental/filesystem>
namespace ns_filesystem = std::experimental::filesystem;


#include "algorithms/nicol1d.hpp"
#include "algorithms/probe_a_load.hpp"
#include "algorithms/refine_a_cut.hpp"
#include "algorithms/uniform.hpp"
#include "data_structures/csr_matrix.hpp"

// using Ordinal = Ordinal;
using Value = Ordinal;
