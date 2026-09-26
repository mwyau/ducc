/*
 *  This code is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 */

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>

namespace ducc0 {
namespace detail_fft {

void sort_factors_descending(std::vector<std::size_t> &factors)
  { std::sort(factors.begin(), factors.end(), std::greater<std::size_t>()); }

} // namespace detail_fft
} // namespace ducc0
