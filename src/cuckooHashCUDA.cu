#include <ctime>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <exception>
#include <omp.h>

#include "cuckooHash.h"
#include "cuckooHashCUDA.h"
#include "input.h"

namespace cuckooHash {
// declare cuda functions
__device__ uint32_t hash_device(size_t i, uint64_t key, uint64_t coef_a,
                                uint64_t coef_b, uint64_t h_array_size);
__global__ void lookup_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint32_t *array_coef_a,
                              uint32_t *array_coef_b, uint32_t *array_key,
                              size_t num_array_key, uint32_t *res_pos_hash_func,
                              uint32_t *res_pos_hash_array);
__global__ void delete_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint32_t *array_coef_a,
                              uint32_t *array_coef_b, uint32_t *array_key,
                              uint32_t num_array_key);
__global__ void insert_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint32_t *array_coef_a,
                              uint32_t *array_coef_b, uint32_t max_eviction,
                              uint32_t *array_key, uint32_t num_array_key,
                              uint32_t *res_pos_hash_func,
                              uint32_t *res_pos_hash_array,
                              bool *exceed_max_eviction);
/**
 * @brief
 *
 * Get them from cuckooHashCUDA:
 * @param hash_table
 * @param size_hash_table
 *
 * Get them from lookup(Instruction inst):
 * @param array_key
 * @param num_aray_key
 *
 * Get them from cuckooHashCUDA:
 * @param array_coef_a
 * @param array_coef_b
 * @param num_hash_func
 *
 * Store the result:
 * @param res_pos_hash_func
 * @param res_pos_hash_array
 * @return __device__
 */
__device__ void lookup_device(uint32_t *hash_table, size_t size_hash_table,
                              uint32_t *array_key, size_t num_array_key,
                              uint32_t *array_coef_a, uint32_t *array_coef_b,
                              size_t num_hash_func, uint32_t *res_pos_hash_func,
                              uint32_t *res_pos_hash_array);

// implement cuda functions
// hash
__device__ uint32_t hash_device(size_t i, uint64_t key, uint64_t coef_a,
                                uint64_t coef_b, uint64_t h_array_size) {
  uint64_t res = (coef_a * key + coef_b) % INT_MAX;
  uint64_t pos = res % h_array_size;
  return pos + i * h_array_size;
}

// lookup
/**
 * @brief
 *
 * Get them from cuckooHashCUDA:
 * @param ptr_hash_table
 * @param size_hash_table
 * @param num_hash_func
 * @param array_coef_a
 * @param array_coef_b
 *
 * Get them from lookup(Instruction inst):
 * @param array_key
 * @param num_array_key
 * @param res_pos_hash_func
 * @param res_pos_hash_array
 * @return __global__
 */
__global__ void lookup_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint32_t *array_coef_a,
                              uint32_t *array_coef_b, uint32_t *array_key,
                              size_t num_array_key, uint32_t *res_pos_hash_func,
                              uint32_t *res_pos_hash_array) {
  lookup_device(ptr_hash_table, size_hash_table, array_key, num_array_key,
                array_coef_a, array_coef_b, num_hash_func, res_pos_hash_func,
                res_pos_hash_array);
}

// delete
__global__ void delete_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint32_t *array_coef_a,
                              uint32_t *array_coef_b, uint32_t *array_key,
                              uint32_t num_array_key) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // lookup the key in the range [i * thread_lookup_size, (i + 1) *
  // thread_lookup_size)
  uint32_t h_array_size = size_hash_table / num_hash_func;
  size_t num_threads = blockDim.x * gridDim.x;
  size_t thread_lookup_size = num_array_key / num_threads;
  thread_lookup_size = thread_lookup_size == 0 ? 1 : thread_lookup_size;

  // go over the array of keys assigned to thread `i`
  size_t array_key_start = i * thread_lookup_size;
  size_t array_key_end = (i + 1) * thread_lookup_size;
  array_key_end = array_key_end > num_array_key ? num_array_key : array_key_end;
  for (size_t k = array_key_start; k < array_key_end; ++k) {
    // for each array corresponding to a hash function
    for (size_t j = 0; j < num_hash_func; ++j) {
      uint32_t pos = hash_device(j, array_key[k], array_coef_a[j],
                                 array_coef_b[j], h_array_size);
      if (ptr_hash_table[pos] == array_key[k]) {
        ptr_hash_table[pos] = 0;
      }
    }
  }
}

// insert
// res_pos_hash_func, res_pos_hash_array are used for lookup
__global__ void insert_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint32_t *array_coef_a,
                              uint32_t *array_coef_b, uint32_t max_eviction,
                              uint32_t *array_key, uint32_t num_array_key,
                              uint32_t *res_pos_hash_func,
                              uint32_t *res_pos_hash_array,
                              bool *exceed_max_eviction) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Insert_global called from thread %d\n", (int)i);

  *exceed_max_eviction = false;
  // lookup
  lookup_device(ptr_hash_table, size_hash_table, array_key, num_array_key,
                array_coef_a, array_coef_b, num_hash_func, res_pos_hash_func,
                res_pos_hash_array);

  // insert the key in the range [i * thread_lookup_size, (i + 1) *
  // thread_lookup_size)
  uint32_t h_array_size = size_hash_table / num_hash_func;
  size_t num_threads = blockDim.x * gridDim.x;
  size_t thread_lookup_size = num_array_key / num_threads;
  thread_lookup_size = thread_lookup_size == 0 ? 1 : thread_lookup_size;

  // go over the array of keys assigned to thread `i`
  size_t array_key_start = i * thread_lookup_size;
  size_t array_key_end = (i + 1) * thread_lookup_size;
  array_key_end = array_key_end > num_array_key ? num_array_key : array_key_end;
  for (size_t k = array_key_start; k < array_key_end; ++k) {
    if (k % 10000 == 0) {
      printf("Inserting %zu\n", k);
    }
    if (res_pos_hash_func[k] != UINT_MAX) {
      continue;
    }
    uint32_t hash_value;
    for (size_t j = 0; j < max_eviction; j += num_hash_func) {
      for (size_t m = 0; m < num_hash_func; m++) {
        // calc hash value
        hash_value = hash_device(m, array_key[k], array_coef_a[m],
                                 array_coef_b[m], h_array_size);
        // try to insert
        // if failed, exchange the key with the existing key
        atomicExch(&ptr_hash_table[hash_value], array_key[k]);
        if (array_key[k] == 0) {
          goto insert_key_end;
        }
      }
    }
    // if exceed the max eviction, set exceed_max_eviction to true
    // then quit the loop
    *exceed_max_eviction |= true;
    return;

  insert_key_end:
    continue;
  }
}

// device functions
__device__ void lookup_device(uint32_t *hash_table, size_t size_hash_table,
                              uint32_t *array_key, size_t num_array_key,
                              uint32_t *array_coef_a, uint32_t *array_coef_b,
                              size_t num_hash_func, uint32_t *res_pos_hash_func,
                              uint32_t *res_pos_hash_array) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // lookup the key in the range [i * thread_lookup_size, (i + 1) *
  // thread_lookup_size)
  uint32_t h_array_size = size_hash_table / num_hash_func;
  size_t num_threads = blockDim.x * gridDim.x;
  size_t thread_lookup_size = num_array_key / num_threads;
  thread_lookup_size = thread_lookup_size == 0 ? 1 : thread_lookup_size;

  // go over the array of keys assigned to thread `i`
  size_t array_key_start = i * thread_lookup_size;
  size_t array_key_end = (i + 1) * thread_lookup_size;
  array_key_end = array_key_end > num_array_key ? num_array_key : array_key_end;
  for (size_t k = array_key_start; k < array_key_end; ++k) {
    res_pos_hash_func[k] = UINT_MAX;
    res_pos_hash_array[k] = UINT_MAX;
    if (array_key[k] == 0) {
      continue;
    }
    // for each array corresponding to a hash function
    for (size_t j = 0; j < num_hash_func; ++j) {
      uint32_t pos = hash_device(j, array_key[k], array_coef_a[j],
                                 array_coef_b[j], h_array_size);
      if (hash_table[pos] == array_key[k]) {
        res_pos_hash_func[k] = i;
        res_pos_hash_array[k] = pos;
      }
    }
  }
}

void CuckooHashCUDA::rehash() {
  std::cout << "Rehashing..." << std::endl;
  auto s_hash_table = get_size_hash_table();
  generate_hash_func_coef();

  uint32_t *ptr_cuda_new_hash_table = nullptr;
  cudaMallocManaged(&ptr_cuda_new_hash_table, s_hash_table * sizeof(uint32_t));
  auto new_hash_table = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_new_hash_table, &cudaMemDeconstructor);
  std::swap(hash_table, new_hash_table);

  // insert
  size_t num_threads = block_size;
  size_t num_blocks = grid_size;
  uint32_t *ptr_cuda_array_key = new_hash_table.get();
  uint32_t *ptr_cuda_res_pos_hash_func = nullptr;
  uint32_t *ptr_cuda_res_pos_hash_array = nullptr;
  bool *ptr_cuda_exceed_max_eviction = nullptr;
  cudaMallocManaged(&ptr_cuda_res_pos_hash_func,
                    s_hash_table * sizeof(uint32_t));
  cudaMallocManaged(&ptr_cuda_res_pos_hash_array,
                    s_hash_table * sizeof(uint32_t));
  cudaMallocManaged(&ptr_cuda_exceed_max_eviction, sizeof(bool));
  auto res_pos_hash_func = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_res_pos_hash_func, &cudaMemDeconstructor);
  auto res_pos_hash_array = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_res_pos_hash_array, &cudaMemDeconstructor);
  auto exceed_max_eviction = std::unique_ptr<bool, cudaMemDeconstructor_t>(
      ptr_cuda_exceed_max_eviction, &cudaMemDeconstructor);

  // get params from cuckooHashCUDA
  uint32_t *ptr_hash_table = hash_table.get();
  size_t size_hash_table = get_size_hash_table();
  size_t num_hash_func = get_num_hash_func();
  uint32_t *array_coef_a = hash_func_coef_a.get();
  uint32_t *array_coef_b = hash_func_coef_b.get();
  uint32_t max_eviction = get_max_eviction();

  while (1) {
    insert_global<<<num_blocks, num_threads>>>(
        ptr_hash_table, size_hash_table, num_hash_func, array_coef_a,
        array_coef_b, max_eviction, ptr_cuda_array_key, s_hash_table,
        res_pos_hash_func.get(), res_pos_hash_array.get(),
        exceed_max_eviction.get());
    cudaDeviceSynchronize();
    if (!(*exceed_max_eviction)) {
      break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    }
    rehash();
  }
}

hashTablePos CuckooHashCUDA::lookup(uint32_t) {
  std::cout << "Do not call lookup(uint32_t key) in CuckooHashCUDA\n";
  return {UINT_MAX, UINT_MAX};
}
std::vector<hashTablePos> CuckooHashCUDA::lookup(Instruction inst) {
  if (inst.first != "lookup") {
    return {};
  }
  size_t num_array_key = inst.second.size();
  uint32_t *ptr_cuda_array_key = nullptr;
  cudaMallocManaged(&ptr_cuda_array_key, num_array_key * sizeof(uint32_t));
  auto array_key = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_array_key, cudaMemDeconstructor);
#pragma omp parallel for
  for (size_t i = 0; i < num_array_key; ++i) {
    array_key[i] = inst.second[i];
  }

  size_t num_threads = block_size;
  size_t num_blocks = grid_size;
  uint32_t *ptr_cuda_res_pos_hash_func = nullptr;
  uint32_t *ptr_cuda_res_pos_hash_array = nullptr;
  cudaMallocManaged(&ptr_cuda_res_pos_hash_func,
                    num_array_key * sizeof(uint32_t));
  cudaMallocManaged(&ptr_cuda_res_pos_hash_array,
                    num_array_key * sizeof(uint32_t));
  auto res_pos_hash_func = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_res_pos_hash_func, &cudaMemDeconstructor);
  auto res_pos_hash_array = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_res_pos_hash_array, &cudaMemDeconstructor);

  lookup_global<<<num_blocks, num_threads>>>(
      hash_table.get(), get_size_hash_table(), get_num_hash_func(),
      hash_func_coef_a.get(), hash_func_coef_b.get(), array_key.get(),
      num_array_key, res_pos_hash_func.get(), res_pos_hash_array.get());
  cudaDeviceSynchronize();

  std::vector<hashTablePos> results;
#pragma omp parallel for
  for (size_t i = 0; i < num_array_key; ++i) {
    results.push_back({res_pos_hash_func[i], res_pos_hash_array[i]});
  }
  return results;
}

void CuckooHashCUDA::delete_key(uint32_t key) {
  hashTablePos pos = lookup(key);
  if (pos == std::make_pair(UINT_MAX, UINT_MAX)) {
    return;
  }

  hash_table[pos.second] = 0;
}

void CuckooHashCUDA::insert(uint32_t key) {
  std::cout << "Do not call insert(uint32_t key) in CuckooHashCUDA\n";
}

void CuckooHashCUDA::insert(Instruction inst) {
  if (inst.first != "insert") {
    return;
  }
  size_t max_eviction = 4 * std::log2(inst.second.size());
  max_eviction = max_eviction < 4 ? 4 : max_eviction;
  set_max_eviction(max_eviction);
  std::cout << "Set max eviction: " << get_max_eviction() << std::endl;

  size_t num_array_key = inst.second.size();
  uint32_t *ptr_cuda_array_key = nullptr;
  cudaMallocManaged(&ptr_cuda_array_key, num_array_key * sizeof(uint32_t));
  auto array_key = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_array_key, &cudaMemDeconstructor);
#pragma omp parallel for
  for (size_t i = 0; i < num_array_key; ++i) {
    array_key[i] = inst.second[i];
  }

  size_t num_threads = block_size;
  size_t num_blocks = grid_size;
  uint32_t *ptr_cuda_res_pos_hash_func = nullptr;
  uint32_t *ptr_cuda_res_pos_hash_array = nullptr;
  bool *ptr_cuda_exceed_max_eviction = nullptr;
  cudaMallocManaged(&ptr_cuda_res_pos_hash_func,
                    num_array_key * sizeof(uint32_t));
  cudaMallocManaged(&ptr_cuda_res_pos_hash_array,
                    num_array_key * sizeof(uint32_t));
  cudaMallocManaged(&ptr_cuda_exceed_max_eviction, sizeof(bool));
  auto res_pos_hash_func = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_res_pos_hash_func, &cudaMemDeconstructor);
  auto res_pos_hash_array = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_res_pos_hash_array, &cudaMemDeconstructor);
  auto exceed_max_eviction = std::unique_ptr<bool, cudaMemDeconstructor_t>(
      ptr_cuda_exceed_max_eviction, &cudaMemDeconstructor);

  // get params from cuckooHashCUDA
  uint32_t *ptr_hash_table = hash_table.get();
  size_t size_hash_table = get_size_hash_table();
  size_t num_hash_func = get_num_hash_func();
  uint32_t *array_coef_a = hash_func_coef_a.get();
  uint32_t *array_coef_b = hash_func_coef_b.get();

#ifdef BENCHMARK
  uint64_t t_start = std::clock();
#endif

  while (1) {
    insert_global<<<num_blocks, num_threads>>>(
        ptr_hash_table, size_hash_table, num_hash_func, array_coef_a,
        array_coef_b, max_eviction, array_key.get(), num_array_key,
        res_pos_hash_func.get(), res_pos_hash_array.get(),
        exceed_max_eviction.get());
    cudaDeviceSynchronize();
    if (!(*exceed_max_eviction)) {
      break;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    }
    rehash();
  }

#ifdef BENCHMARK
  uint64_t t_end = std::clock();
  std::cout << "Time elapsed in inserting " << inst.second.size()
            << " items: " << t_end - t_start << "\n";
#endif
#ifdef DEBUG
  println("Final hash table:");
  print();
#endif
}

void CuckooHashCUDA::print() {
  std::cout << "Do not call print() in CuckooHashCUDA\n";
}
} // namespace cuckooHash