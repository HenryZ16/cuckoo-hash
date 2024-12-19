#pragma once

#include <memory>

#include "config.h"
#include "cuckooHash.h"

namespace cuckooHash {
class CuckooHashDefault : public CuckooHash {
  void rehash() override;
  std::vector<hashFuncCoef> hash_func_coef;
  void generate_hash_func_coef() {
    std::random_device rd;
    std::mt19937 gen(rd());
    size_t num_hash_func = get_num_hash_func();
    size_t size_hash_table = get_size_hash_table();
    size_t rd_upper = size_hash_table >> 4;
    rd_upper = rd_upper < 64 ? 64 : rd_upper;
    std::uniform_int_distribution<> dis(1, rd_upper);
    for (size_t i = 0; i < num_hash_func; i++) {
      hash_func_coef[i] = {dis(gen), dis(gen)};
    }
  }
  uint32_t hash(uint32_t key, size_t i) const {
    size_t num_hash_func = get_num_hash_func();
    size_t size_hash_table = get_size_hash_table();
    uint64_t u64_key = key;
    uint64_t a = hash_func_coef[i].first;
    uint64_t b = hash_func_coef[i].second;
    uint64_t res = (a * u64_key + b) % INT_MAX;
    uint64_t h_array_size = size_hash_table / num_hash_func;
    uint64_t pos = res % h_array_size;
    return pos + i * h_array_size;
  }

  // protected:
  //  // num_hash_func * size_hash_table
  //  std::unique_ptr<uint32_t[]> hash_table;
  //  std::vector<hashFuncCoef> hash_func_coef;
  std::unique_ptr<uint32_t[]> hash_table;

public:
  // Constructors
  CuckooHashDefault() = delete;
  CuckooHashDefault(const size_t num_hash_func, const size_t size_hash_table)
      : CuckooHash(num_hash_func, size_hash_table),
        hash_func_coef(num_hash_func),
        hash_table(new uint32_t[size_hash_table]) {
    generate_hash_func_coef();
  }
  CuckooHashDefault(const Config &config)
      : CuckooHash(config.get_num_hash_func(), config.get_size_hash_table()),
        hash_func_coef(config.get_num_hash_func()),
        hash_table(new uint32_t[config.get_size_hash_table()]) {
    generate_hash_func_coef();
  }

  // Default destructor, copy constructor, and copy assignment operator
  // Operations
  virtual void insert(uint32_t key) override;
  virtual void insert(const Instruction& inst) override;
  virtual hashTablePos lookup(uint32_t key) override;
  virtual void lookup(const Instruction& inst,
                      std::vector<hashTablePos> &res) override;
  virtual void delete_key(uint32_t key) override;
  virtual void print() override;
  virtual ~CuckooHashDefault() = default;
};
} // namespace cuckooHash