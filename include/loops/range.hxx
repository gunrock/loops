/**
 * @file range.hxx
 * @brief Code is based on Mark Harris' work:
 * https://github.com/harrism/cpp11-range/blob/master/range.hpp
 * @version 0.1
 * @date 2022-02-02
 *
 * @copyright Copyright (c) 2022
 *3
 */
#pragma once

#include <iterator>
#include <type_traits>

namespace loops {
namespace detail {

template <typename type_t>
struct range_iter_base : std::iterator<std::input_iterator_tag, type_t> {
  __host__ __device__ range_iter_base(type_t current) : current(current) {}

  __host__ __device__ type_t operator*() const { return current; }

  __host__ __device__ type_t const* operator->() const { return &current; }

  __host__ __device__ range_iter_base& operator++() {
    ++current;
    return *this;
  }

  __host__ __device__ range_iter_base operator++(int) {
    auto copy = *this;
    ++*this;
    return copy;
  }

  __host__ __device__ bool operator==(range_iter_base const& other) const {
    return current == other.current;
  }

  __host__ __device__ bool operator!=(range_iter_base const& other) const {
    return !(*this == other);
  }

 protected:
  type_t current;
};

}  // namespace detail

template <typename type_t>
struct range_proxy {
  struct iter : detail::range_iter_base<type_t> {
    __host__ __device__ iter(type_t current)
        : detail::range_iter_base<type_t>(current) {}
  };

  struct step_range_proxy {
    struct iter : detail::range_iter_base<type_t> {
      __host__ __device__ iter(type_t current, type_t step)
          : detail::range_iter_base<type_t>(current), step(step) {}

      using detail::range_iter_base<type_t>::current;

      __host__ __device__ iter& operator++() {
        current += step;
        return *this;
      }

      __host__ __device__ iter operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
      }

      // Loses commutativity. Iterator-based ranges are simply broken.
      __host__ __device__ bool operator==(iter const& other) const {
        return step > 0 ? current >= other.current : current < other.current;
      }

      __host__ __device__ bool operator!=(iter const& other) const {
        return !(*this == other);
      }

     private:
      type_t step;
    };

    __host__ __device__ step_range_proxy(type_t begin, type_t end, type_t step)
        : begin_(begin, step), end_(end, step) {}

    __host__ __device__ iter begin() const { return begin_; }

    __host__ __device__ iter end() const { return end_; }

   private:
    iter begin_;
    iter end_;
  };

  __host__ __device__ range_proxy(type_t begin, type_t end)
      : begin_(begin), end_(end) {}

  __host__ __device__ step_range_proxy step(type_t step) {
    return {*begin_, *end_, step};
  }

  __host__ __device__ iter begin() const { return begin_; }

  __host__ __device__ iter end() const { return end_; }

 private:
  iter begin_;
  iter end_;
};

template <typename type_t>
struct infinite_range_proxy {
  struct iter : detail::range_iter_base<type_t> {
    __host__ __device__ iter(type_t current = type_t())
        : detail::range_iter_base<type_t>(current) {}

    __host__ __device__ bool operator==(iter const&) const { return false; }

    __host__ __device__ bool operator!=(iter const&) const { return true; }
  };

  struct step_range_proxy {
    struct iter : detail::range_iter_base<type_t> {
      __host__ __device__ iter(type_t current = type_t(),
                               type_t step = type_t())
          : detail::range_iter_base<type_t>(current), step(step) {}

      using detail::range_iter_base<type_t>::current;

      __host__ __device__ iter& operator++() {
        current += step;
        return *this;
      }

      __host__ __device__ iter operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
      }

      __host__ __device__ bool operator==(iter const&) const { return false; }

      __host__ __device__ bool operator!=(iter const&) const { return true; }

     private:
      type_t step;
    };

    __host__ __device__ step_range_proxy(type_t begin, type_t step)
        : begin_(begin, step) {}

    __host__ __device__ iter begin() const { return begin_; }

    __host__ __device__ iter end() const { return iter(); }

   private:
    iter begin_;
  };

  __host__ __device__ infinite_range_proxy(type_t begin) : begin_(begin) {}

  __host__ __device__ step_range_proxy step(type_t step) {
    return step_range_proxy(*begin_, step);
  }

  __host__ __device__ iter begin() const { return begin_; }

  __host__ __device__ iter end() const { return iter(); }

 private:
  iter begin_;
};

template <typename type_t>
__host__ __device__ range_proxy<type_t> range(type_t begin, type_t end) {
  return {begin, end};
}

template <typename type_t>
__host__ __device__ infinite_range_proxy<type_t> range(type_t begin) {
  return {begin};
}

namespace traits {

template <typename C>
struct has_size {
  template <typename type_t>
  static constexpr auto check(type_t*) -> typename std::is_integral<
      decltype(std::declval<type_t const>().size())>::type;

  template <typename>
  static constexpr auto check(...) -> std::false_type;

  using type = decltype(check<C>(0));
  static constexpr bool value = type::value;
};

}  // namespace traits

template <typename C,
          typename = typename std::enable_if<traits::has_size<C>::value>>
__host__ __device__ auto indices(C const& cont)
    -> range_proxy<decltype(cont.size())> {
  return {0, cont.size()};
}

template <typename type_t, std::size_t N>
__host__ __device__ range_proxy<std::size_t> indices(type_t (&)[N]) {
  return {0, N};
}

template <typename type_t>
range_proxy<typename std::initializer_list<type_t>::size_type>
    __host__ __device__ indices(std::initializer_list<type_t>&& cont) {
  return {0, cont.size()};
}
}  // namespace loops