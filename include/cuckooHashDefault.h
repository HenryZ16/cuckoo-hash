#pragma once

#include <memory>

#include "config.h"
#include "cuckooHash.h"

namespace cuckooHash {
class CuckooHashDefault : public CuckooHash {
  void rehash() override;

  // protected:
  //  // num_hash_func * size_hash_table
  //  std::vector<std::unique_ptr<uint32_t>> hash_table;
  //  std::vector<hashFuncCoef> hash_func_coef;

public:
  // Constructors
  CuckooHashDefault() = delete;
  CuckooHashDefault(const size_t num_hash_func, const size_t size_hash_table)
      : CuckooHash(num_hash_func, size_hash_table) {
    generate_hash_func_coef();
  }
  CuckooHashDefault(const Config &config)
      : CuckooHash(config.get_num_hash_func(), config.get_size_hash_table()) {
    generate_hash_func_coef();
  }

  // Default destructor, copy constructor, and copy assignment operator
  // Operations
  virtual void insert(uint32_t key) override;
  virtual void insert(Instruction inst) override;
  virtual hashTablePos lookup(uint32_t key) override;
  virtual void delete_key(uint32_t key) override;
  virtual void print() override;
  virtual ~CuckooHashDefault() = default;
};
} // namespace cuckooHash