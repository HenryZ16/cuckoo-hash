#pragma once

#include <memory>

#include "cuckooHash.h"

namespace cuckooHash {
class CuckooHashDefault : public CuckooHash {
  void rehash() override;

  // protected:
  //  // num_hash_func * size_hash_table
  //  std::vector<std::unique_ptr<int>> hash_table;
  //  std::vector<hashFuncCoef> hash_func_coef;

public:
  // Constructors
  CuckooHashDefault() = delete;
  CuckooHashDefault(const size_t num_hash_func, const size_t size_hash_table)
      : CuckooHash(num_hash_func, size_hash_table) {
    for (auto &uptr : hash_table) {
      uptr = std::make_unique<int[]>(size_hash_table);
    }
    generate_hash_func_coef();
  }

  // Default destructor, copy constructor, and copy assignment operator
  // Operations
  virtual void insert(int key) override;
  virtual void insert(Instruction inst) override;
  virtual hashTablePos lookup(int key) override;
  virtual void delete_key(int key) override;
  virtual void print() override;
  virtual ~CuckooHashDefault() = default;
};
} // namespace cuckooHash