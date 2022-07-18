/**
 * @file helpers.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Header file for SpMV.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <loops/schedule.hxx>

#include <loops/util/generate.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/container/market.hxx>
#include <loops/util/filepath.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/equal.hxx>
#include <loops/util/device.hxx>
#include <loops/util/timer.hxx>
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
      : options(argv[0], "Sparse Matrix-Vector Multiplication") {
    // Add command line options
    options.add_options()("h,help", "Print help")                   // help
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

using namespace loops;
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
loops::vector_t<type_t, memory_space_t::host> reference(
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

/**
 * @brief Validation for SpMV.
 *
 * @tparam index_t Column indices type.
 * @tparam offset_t Row offset type.
 * @tparam type_t Value type.
 * @param parameters Parameters.
 * @param csr CSR matrix.
 * @param x Input vector.
 * @param y Output vector.
 */
template <typename index_t, typename offset_t, typename type_t>
void validate(parameters_t& parameters,
              csr_t<index_t, offset_t, type_t>& csr,
              vector_t<type_t>& x,
              vector_t<type_t>& y) {
  // Validation code, can be safely ignored.
  auto h_y = reference(csr, x);

  std::size_t errors = util::equal(
      y.data().get(), h_y.data(), csr.rows,
      [](const type_t a, const type_t b) { return std::abs(a - b) > 1e-2; },
      parameters.verbose);

  std::cout << "Matrix:\t\t" << extract_filename(parameters.filename)
            << std::endl;
  std::cout << "Dimensions:\t" << csr.rows << " x " << csr.cols << " ("
            << csr.nnzs << ")" << std::endl;
  std::cout << "Errors:\t\t" << errors << std::endl;
}

}  // namespace cpu