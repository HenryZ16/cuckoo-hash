#pragma once

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "input.h"

namespace cuckooHash {
typedef std::pair<uint32_t, uint32_t> hashFuncCoef;
typedef std::pair<uint32_t, uint32_t> hashTablePos;
// An abstract class for Cuckoo Hash
class CuckooHash {
  size_t num_hash_func;
  size_t size_hash_table;
  size_t max_eviction;

protected:
  // num_hash_func * size_hash_table
  std::unique_ptr<uint32_t[]> hash_table;
  std::vector<hashFuncCoef> hash_func_coef;
  virtual void rehash() = 0;
  void generate_hash_func_coef() {
    std::random_device rd;
    std::mt19937 gen(rd());
    size_t rd_upper = size_hash_table >> 4;
    rd_upper = rd_upper < 64 ? 64 : rd_upper;
    std::uniform_int_distribution<> dis(1, rd_upper);
    for (size_t i = 0; i < num_hash_func; i++) {
      hash_func_coef[i] = {dis(gen), dis(gen)};
    }
  }
  uint32_t hash(uint32_t key, size_t i) const {
    uint64_t u64_key = key;
    uint64_t a = hash_func_coef[i].first;
    uint64_t b = hash_func_coef[i].second;
    uint64_t res = (a * u64_key + b) % UINT_MAX;
    uint64_t h_array_size = size_hash_table / num_hash_func;
    uint64_t pos = res % h_array_size;
    return pos + i * h_array_size;
  }

public:
  // Constructors
  CuckooHash() = delete;
  CuckooHash(const size_t num_hash_func, const size_t size_hash_table)
      : num_hash_func(num_hash_func), size_hash_table(size_hash_table),
        max_eviction(100), hash_table(new uint32_t[size_hash_table]),
        hash_func_coef(num_hash_func) {}

  // Default destructor, copy constructor, and copy assignment operator
  // member Accessors
  size_t get_num_hash_func() const { return num_hash_func; }
  size_t get_size_hash_table() const { return size_hash_table; }
  size_t get_max_eviction() const { return max_eviction; }
  void set_max_eviction(size_t max_eviction) {
    this->max_eviction = max_eviction;
  }

  // Operations
  virtual void insert(uint32_t key) = 0;
  virtual void insert(Instruction inst) = 0;
  virtual hashTablePos lookup(uint32_t key) = 0;
  virtual void delete_key(uint32_t key) = 0;
  virtual void print() = 0;
  virtual ~CuckooHash() = default;
};
} // namespace cuckooHash