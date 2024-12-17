#pragma once

#include <memory>

#include "config.h"
#include "cuckooHash.h"

/* 1. calc hash_value for all keys
   2. insert all keys to hash table
   3. wait for 2 to finish
   4. check if all keys are inserted successfully
   5. if 4 failed, go to 1. Else finish.
*/

#pragma once

#include <cuda_runtime.h>
#include <memory>

#include "config.h"
#include "cuckooHash.h"

namespace cuckooHash {
static const uint32_t SM_2080Ti = 68;
static const uint32_t SM_V100 = 80;
class CuckooHashCUDA : public CuckooHash {
  void rehash() override;
  static void cudaMemDeconstructor(void *ptr) { cudaFree(ptr); }
  typedef decltype(&cudaMemDeconstructor) cudaMemDeconstructor_t;

  std::unique_ptr<uint32_t[], cudaMemDeconstructor_t> hash_func_coef_a;
  std::unique_ptr<uint32_t[], cudaMemDeconstructor_t> hash_func_coef_b;
  void generate_hash_func_coef() {
    std::random_device rd;
    std::mt19937 gen(rd());
    size_t num_hash_func = get_num_hash_func();
    size_t size_hash_table = get_size_hash_table();
    size_t rd_upper = size_hash_table >> 4;
    rd_upper = rd_upper < 64 ? 64 : rd_upper;
    std::uniform_int_distribution<> dis(1, rd_upper);
    for (size_t i = 0; i < num_hash_func; i++) {
      hash_func_coef_a[i] = dis(gen);
      hash_func_coef_b[i] = dis(gen);
    }
  }

  // protected:
  //  // num_hash_func * size_hash_table
  //  std::unique_ptr<uint32_t[]> hash_table;
  //  std::vector<hashFuncCoef> hash_func_coef;
  std::unique_ptr<uint32_t[], cudaMemDeconstructor_t> hash_table;
  uint64_t block_size;
  uint64_t grid_size;

public:
  // Constructors
  CuckooHashCUDA() = delete;
  CuckooHashCUDA(const size_t num_hash_func, const size_t size_hash_table)
      : CuckooHash(num_hash_func, size_hash_table),
        hash_func_coef_a(nullptr, &cudaMemDeconstructor),
        hash_func_coef_b(nullptr, &cudaMemDeconstructor),
        hash_table(nullptr, &cudaMemDeconstructor), block_size(256),
        grid_size(256) {
    uint32_t *ptr_cuda_hash_table = nullptr;
    uint32_t *ptr_cuda_hash_func_coef_a = nullptr;
    uint32_t *ptr_cuda_hash_func_coef_b = nullptr;
    cudaMalloc(&ptr_cuda_hash_table, size_hash_table * sizeof(uint32_t));
    cudaMallocManaged(&ptr_cuda_hash_func_coef_a,
                      num_hash_func * sizeof(uint32_t));
    cudaMallocManaged(&ptr_cuda_hash_func_coef_b,
                      num_hash_func * sizeof(uint32_t));
    hash_table.reset(ptr_cuda_hash_table);
    hash_func_coef_a.reset(ptr_cuda_hash_func_coef_a);
    hash_func_coef_b.reset(ptr_cuda_hash_func_coef_b);
    generate_hash_func_coef();
  }
  CuckooHashCUDA(const Config &config)
      : CuckooHash(config.get_num_hash_func(), config.get_size_hash_table()),
        hash_func_coef_a(nullptr, &cudaMemDeconstructor),
        hash_func_coef_b(nullptr, &cudaMemDeconstructor),
        hash_table(nullptr, &cudaMemDeconstructor), block_size(256),
        grid_size(256) {
    size_t size_hash_table = get_size_hash_table();
    size_t num_hash_func = get_num_hash_func();
    uint32_t *ptr_cuda_hash_table = nullptr;
    uint32_t *ptr_cuda_hash_func_coef_a = nullptr;
    uint32_t *ptr_cuda_hash_func_coef_b = nullptr;
    cudaMalloc(&ptr_cuda_hash_table, size_hash_table * sizeof(uint32_t));
    cudaMallocManaged(&ptr_cuda_hash_func_coef_a,
                      num_hash_func * sizeof(uint32_t));
    cudaMallocManaged(&ptr_cuda_hash_func_coef_b,
                      num_hash_func * sizeof(uint32_t));
    hash_table.reset(ptr_cuda_hash_table);
    hash_func_coef_a.reset(ptr_cuda_hash_func_coef_a);
    hash_func_coef_b.reset(ptr_cuda_hash_func_coef_b);
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