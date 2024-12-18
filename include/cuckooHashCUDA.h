#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <memory>

#include "config.h"
#include "cuckooHash.h"
#include "primeFilter.h"

#define BLOCK_SIZE 32
#define GRID_SIZE 2048
#define PRIME INT_MAX
#define NUM_HASH_FUNC_COEF 4

/* 1. calc hash_value for all keys
   2. insert all keys to hash table
   3. wait for 2 to finish
   4. check if all keys are inserted successfully
   5. if 4 failed, go to 1. Else finish.
*/

namespace cuckooHash {
static const uint32_t SM_2080Ti = 68;
static const uint32_t SM_V100 = 80;
class CuckooHashCUDA : public CuckooHash {
  void rehash() override;
  static void cudaMemDeconstructor(void *ptr) { cudaFree(ptr); }
  typedef decltype(&cudaMemDeconstructor) cudaMemDeconstructor_t;

  // num_hash_func * NUM_HASH_FUNC_COEF
  std::unique_ptr<uint64_t[], cudaMemDeconstructor_t> hash_func_coef;
  void generate_hash_func_coef() {
    std::random_device rd;
    std::mt19937 gen(rd());
    size_t num_hash_func = get_num_hash_func();
    size_t size_hash_table = get_size_hash_table();
    std::uniform_int_distribution<> dis(63, prime_cnt);
    std::cout << "Generated hash function coefficients: ";
    for (size_t i = 0; i < num_hash_func * NUM_HASH_FUNC_COEF; i++) {
      hash_func_coef[i] =
          prime_list[dis(gen)] * prime_list[dis(gen)] + prime_list[dis(gen)];
      std::cout << hash_func_coef[i] << ", ";
    }
    std::cout << std::endl;
  }

  // protected:
  //  // num_hash_func * size_hash_table
  //  std::unique_ptr<uint32_t[]> hash_table;
  //  std::vector<hashFuncCoef> hash_func_coef;
  std::unique_ptr<uint32_t[], cudaMemDeconstructor_t> hash_table;
  std::unique_ptr<uint32_t[]> prime_list;
  uint64_t block_size;
  uint64_t grid_size;
  uint32_t prime_cnt;

public:
  // Constructors
  CuckooHashCUDA() = delete;
  CuckooHashCUDA(const size_t num_hash_func, const size_t size_hash_table)
      : CuckooHash(num_hash_func, size_hash_table),
        hash_func_coef(nullptr, &cudaMemDeconstructor),
        hash_table(nullptr, &cudaMemDeconstructor),
        prime_list(new uint32_t[size_hash_table >> 2]), block_size(BLOCK_SIZE),
        grid_size(GRID_SIZE), prime_cnt(0) {
    uint32_t *ptr_cuda_hash_table = nullptr;
    uint64_t *ptr_cuda_hash_func_coef = nullptr;
    cudaMallocManaged(&ptr_cuda_hash_table, size_hash_table * sizeof(uint32_t));
    cudaMallocManaged(&ptr_cuda_hash_func_coef,
                      num_hash_func * NUM_HASH_FUNC_COEF * sizeof(uint64_t));
    hash_table.reset(ptr_cuda_hash_table);
    hash_func_coef.reset(ptr_cuda_hash_func_coef);
    prime_cnt = prime_filter(prime_list.get(), size_hash_table >> 5);
    generate_hash_func_coef();
  }
  CuckooHashCUDA(const Config &config)
      : CuckooHash(config.get_num_hash_func(), config.get_size_hash_table()),
        hash_func_coef(nullptr, &cudaMemDeconstructor),
        hash_table(nullptr, &cudaMemDeconstructor),
        prime_list(new uint32_t[config.get_size_hash_table() >> 2]),
        block_size(BLOCK_SIZE), grid_size(GRID_SIZE), prime_cnt(0) {
    size_t size_hash_table = get_size_hash_table();
    size_t num_hash_func = get_num_hash_func();
    uint32_t *ptr_cuda_hash_table = nullptr;
    uint64_t *ptr_cuda_hash_func_coef = nullptr;
    cudaMallocManaged(&ptr_cuda_hash_table, size_hash_table * sizeof(uint32_t));
    cudaMallocManaged(&ptr_cuda_hash_func_coef,
                      num_hash_func * NUM_HASH_FUNC_COEF * sizeof(uint64_t));
    hash_table.reset(ptr_cuda_hash_table);
    hash_func_coef.reset(ptr_cuda_hash_func_coef);
    prime_cnt = prime_filter(prime_list.get(), size_hash_table >> 5);
    generate_hash_func_coef();
  }

  // Default destructor, copy constructor, and copy assignment operator
  // Operations
  virtual void insert(uint32_t key) override;
  virtual void insert(Instruction inst) override;
  virtual hashTablePos lookup(uint32_t key) override;
  virtual std::vector<hashTablePos> lookup(Instruction inst) override;
  virtual void delete_key(uint32_t key) override;
  virtual void print() override;
  virtual ~CuckooHashCUDA() = default;
};
} // namespace cuckooHash