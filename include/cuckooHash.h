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
  virtual void rehash() = 0;

public:
  // Constructors
  CuckooHash() = delete;
  CuckooHash(const size_t num_hash_func, const size_t size_hash_table)
      : num_hash_func(num_hash_func), size_hash_table(size_hash_table),
        max_eviction(100) {}

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
  virtual std::vector<hashTablePos> lookup(Instruction inst) = 0;
  virtual void delete_key(uint32_t key) = 0;
  virtual void print() = 0;
  virtual ~CuckooHash() = default;
};
} // namespace cuckooHash