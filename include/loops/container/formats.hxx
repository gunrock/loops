/**
 * @file formats.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <loops/memory.hxx>

namespace loops {

using namespace memory;

// Forward decleration
template <typename index_t, typename value_t, memory_space_t space>
struct coo_t;

template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space>
struct csr_t;

template <typename index_t,
          typename offset_t,
          typename value_t,
          memory_space_t space>
struct csc_t;

}  // namespace loops

#include <loops/container/coo.hxx>
#include <loops/container/csc.hxx>
#include <loops/container/csr.hxx>