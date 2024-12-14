#pragma once

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "input.h"

namespace cuckooHash {
typedef std::pair<int, int> hashFuncCoef;
typedef std::pair<int, int> hashTablePos;
// An abstract class for Cuckoo Hash
class CuckooHash {
  size_t num_hash_func;
  size_t size_hash_table;
  size_t max_eviction;

protected:
  // num_hash_func * size_hash_table
  std::vector<std::unique_ptr<int[]>> hash_table;
  std::vector<hashFuncCoef> hash_func_coef;
  virtual void rehash() = 0;
  void generate_hash_func_coef() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    for (size_t i = 0; i < num_hash_func; i++) {
      hash_func_coef[i] = {dis(gen), dis(gen)};
    }
  }
  int hash(int key, int i) const {
    return (hash_func_coef[i].first * key + hash_func_coef[i].second) %
           size_hash_table;
  }

public:
  // Constructors
  CuckooHash() = delete;
  CuckooHash(const size_t num_hash_func, const size_t size_hash_table)
      : num_hash_func(num_hash_func), size_hash_table(size_hash_table),
        max_eviction(100), hash_table(num_hash_func),
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
  virtual void insert(int key) = 0;
  virtual void insert(Instruction inst) = 0;
  virtual hashTablePos lookup(int key) = 0;
  virtual void delete_key(int key) = 0;
  virtual void print() = 0;
  virtual ~CuckooHash() = default;
};
} // namespace cuckooHash