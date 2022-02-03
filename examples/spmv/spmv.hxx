/**
 * @file spmv.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Header file for SpMV.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/grid_stride_range.hxx>
#include <loops/util/generate.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/container/market.hxx>
#include <loops/util/filepath.hxx>
#include <loops/memory.hxx>
#include <cxxopts.hpp>

#include <algorithm>
#include <iostream>

struct parameters_t {
  std::string filename;
  bool validate;
  bool verbose;
  cxxopts::Options options;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv)
      : options(argv[0], "Sparse Matrix-Vector Multiplication example") {
    // Add command line options
    options.add_options()("help", "Print help")                     // help
        ("m,market", "Matrix file", cxxopts::value<std::string>())  // mtx
        ("validate", "CPU validation")                              // validate
        ("v,verbose", "Verbose output");                            // verbose

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help") || (result.count("market") == 0)) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("market") == 1) {
      filename = result["market"].as<std::string>();
      if (loops::is_market(filename)) {
      } else {
        std::cout << options.help({""}) << std::endl;
        std::exit(0);
      }
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("validate") == 1) {
      validate = true;
    } else {
      validate = false;
    }

    if (result.count("verbose") == 1) {
      verbose = true;
    } else {
      verbose = false;
    }
  }
};

namespace cpu {

using namespace loops::memory;

/**
 * @brief CPU SpMV implementation.
 *
 * @tparam index_t
 * @tparam offset_t
 * @tparam type_t
 * @param csr device CSR matrix.
 * @param x device input vector.
 * @return loops::vector_t<type_t, memory_space_t::host> device output vector.
 */
template <typename index_t, typename offset_t, typename type_t>
loops::vector_t<type_t, memory_space_t::host> spmv(
    loops::csr_t<index_t, offset_t, type_t, memory_space_t::device>& csr,
    loops::vector_t<type_t, memory_space_t::device>& x) {
  // Copy data to CPU.
  loops::csr_t<index_t, offset_t, type_t, memory_space_t::host> csr_h(csr);
  loops::vector_t<type_t, memory_space_t::host> x_h(x);
  loops::vector_t<type_t, memory_space_t::host> y_h(x_h.size());

  for (auto row = 0; row < csr_h.rows; ++row) {
    type_t sum = 0;
    for (auto nz = csr_h.offsets[row]; nz < csr_h.offsets[row + 1]; ++nz) {
      sum += csr_h.values[nz] * x_h[csr_h.indices[nz]];
    }
    y_h[row] = sum;
  }

  return y_h;
}
}  // namespace cpu