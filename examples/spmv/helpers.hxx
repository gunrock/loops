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

#include <loops/util/generate.hxx>
#include <loops/util/reference.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/container/market.hxx>
#include <loops/util/filepath.hxx>
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
 * @brief Example-side validation: compute the host CSR reference, count
 * mismatches against the GPU result, and print a one-line summary.
 *
 * Both the reference computation and the mismatch counting live in
 * @c loops::reference (see @c include/loops/util/reference.hxx ); this
 * thin wrapper exists so the example binaries can keep the pretty
 * "Matrix / Dimensions / Errors" stdout block the test harness scrapes.
 */
template <typename index_t, typename offset_t, typename type_t>
void validate(parameters_t& parameters,
              csr_t<index_t, offset_t, type_t>& csr,
              vector_t<type_t>& x,
              vector_t<type_t>& y) {
  auto h_y = loops::reference::spmv(csr, x);
  std::size_t errors = loops::reference::count_errors(
      y.data().get(), h_y.data(), csr.rows, parameters.verbose);

  std::cout << "Matrix:\t\t" << extract_filename(parameters.filename)
            << std::endl;
  std::cout << "Dimensions:\t" << csr.rows << " x " << csr.cols << " ("
            << csr.nnzs << ")" << std::endl;
  std::cout << "Errors:\t\t" << errors << std::endl;
}

}  // namespace cpu