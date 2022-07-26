#pragma once

#include <loops/container/vector.hxx>
#include <loops/memory.hxx>

namespace loops {

template <typename value_t, memory_space_t space = memory_space_t::device>
struct matrix_t {
  std::size_t rows;
  std::size_t cols;

  vector_t<value_t, space> m_data;
  value_t* m_data_ptr;

  matrix_t() : rows(0), cols(0), m_data(), m_data_ptr(nullptr) {}

  matrix_t(std::size_t r, std::size_t c)
      : rows(r),
        cols(c),
        m_data(r * c),
        m_data_ptr(memory::raw_pointer_cast(m_data.data())) {}

  __host__ __device__ matrix_t(const matrix_t<value_t, space>& other)
      : rows(other.rows), cols(other.cols), m_data_ptr(other.m_data_ptr) {}

  __host__ __device__ __forceinline__ value_t operator()(int r, int c) const {
    std::size_t idx = (cols * r) + c;
    return m_data_ptr[idx];
  }

  __host__ __device__ __forceinline__ value_t& operator()(int r, int c) {
    std::size_t idx = (cols * r) + c;
    return m_data_ptr[idx];
  }

  __host__ __device__ __forceinline__ value_t
  operator[](std::size_t index) const {
    std::size_t r = index / cols;
    std::size_t c = index % cols;
    std::size_t idx = (cols * r) + c;
    return m_data_ptr[idx];
  }

  __host__ __device__ __forceinline__ value_t& operator[](std::size_t index) {
    std::size_t r = index / cols;
    std::size_t c = index % cols;
    std::size_t idx = (cols * r) + c;
    return m_data_ptr[idx];
  }
};

}  // namespace loops