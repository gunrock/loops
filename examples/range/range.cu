#include <iostream>
#include <vector>

#include <loops/range.hxx>

using namespace loops;

int main() {
  for (auto i : range(1, 5))
    std::cout << i << std::endl;

  for (auto u : range(0u))
    if (u == 3u)
      break;
    else
      std::cout << u << std::endl;

  for (auto c : range('a', 'd'))
    std::cout << c << std::endl;

  for (auto u : range(20u, 29u).step(2u))
    std::cout << u << std::endl;

  for (auto i : range(100).step(-3))
    if (i < 90)
      break;
    else
      std::cout << i << std::endl;

  std::vector<int> x{1, 2, 3};
  for (auto i : indices(x))
    std::cout << i << std::endl;

  for (auto i : indices({"foo", "bar"}))
    std::cout << i << std::endl;

  for (auto i : indices("foobar").step(2))
    std::cout << i << std::endl;
}