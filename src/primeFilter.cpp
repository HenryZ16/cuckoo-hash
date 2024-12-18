#include <memory>
#include <omp.h>

#include "primeFilter.h"

uint32_t prime_filter(uint32_t *array, uint32_t n) {
  uint32_t cnt = 0;
  auto is_prime = std::unique_ptr<bool[]>(new bool[n + 1]);
#pragma omp parallel for
  for (uint32_t i = 2; i <= n; i++) {
    is_prime[i] = true;
  }

#pragma omp parallel for
  for (uint32_t i = 2; i <= n; ++i) {
    if (is_prime[i]) {
      for (uint32_t j = i << 1; j <= n; j += i) {
        is_prime[j] = false;
      }
    }
  }

  for (uint32_t i = 2; i <= n; i++) {
    if (is_prime[i]) {
      array[cnt] = i;
      cnt++;
    }
  }

  return cnt;
}