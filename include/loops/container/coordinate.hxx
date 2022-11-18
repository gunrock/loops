/**
 * @file coordinate.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Simple coordinate with x and y.
 * @version 0.1
 * @date 2022-11-12
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once
namespace loops {
template <typename index_t>
struct coordinate_t {
  index_t x;
  index_t y;
};
}  // namespace loops