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
__device__ uint32_t hash_device(size_t i, uint64_t key, uint64_t *coef,
                                uint64_t h_array_size);
__global__ void lookup_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint64_t *array_coef,
                              uint32_t *array_key, size_t num_array_key,
                              uint32_t *res_pos_hash_func,
                              uint32_t *res_pos_hash_array);
__global__ void delete_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint64_t *array_coef,
                              uint32_t *array_key, uint32_t num_array_key);
__global__ void insert_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint64_t *array_coef,
                              uint32_t max_eviction, uint32_t *array_key,
                              uint32_t num_array_key,
                              uint32_t *exceed_max_eviction,
                              uint32_t if_replace_key);
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
 * @param array_coef
 * @param num_hash_func
 *
 * Store the result:
 * @param res_pos_hash_func
 * @param res_pos_hash_array
 * @return __device__
 */
__device__ void lookup_device(uint32_t *hash_table, size_t size_hash_table,
                              uint32_t *array_key, size_t num_array_key,
                              uint64_t *array_coef, size_t num_hash_func,
                              uint32_t *res_pos_hash_func,
                              uint32_t *res_pos_hash_array);

// implement cuda functions
// hash
__device__ uint32_t hash_device(size_t i, uint64_t key, uint64_t *coef,
                                uint64_t h_array_size) {
  uint64_t value_1 = 0;
#pragma unroll
  for (int i = 0; i < NUM_HASH_FUNC_COEF >> 1; ++i) {
    value_1 *= key;
    value_1 += coef[i];
  }

  uint64_t value_2 = 0;
#pragma unroll
  for (int i = NUM_HASH_FUNC_COEF >> 1; i < NUM_HASH_FUNC_COEF; ++i) {
    value_2 *= key;
    value_2 += coef[i];
  }

  uint64_t value = (value_1 ^ value_2) % PRIME;
  uint64_t pos = value % h_array_size;
  uint32_t ret = static_cast<uint32_t>(pos + i * h_array_size);
  return ret;
}

// lookup
/**
 * @brief
 *
 * Get them from cuckooHashCUDA:
 * @param ptr_hash_table
 * @param size_hash_table
 * @param num_hash_func
 * @param array_coef
 *
 * Get them from lookup(Instruction inst):
 * @param array_key
 * @param num_array_key
 * @param res_pos_hash_func
 * @param res_pos_hash_array
 * @return __global__
 */
__global__ void lookup_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint64_t *array_coef,
                              uint32_t *array_key, size_t num_array_key,
                              uint32_t *res_pos_hash_func,
                              uint32_t *res_pos_hash_array) {
  lookup_device(ptr_hash_table, size_hash_table, array_key, num_array_key,
                array_coef, num_hash_func, res_pos_hash_func,
                res_pos_hash_array);
}

// delete
__global__ void delete_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint64_t *array_coef,
                              uint32_t *array_key, uint32_t num_array_key) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // lookup the key in the range [i * thread_lookup_size, (i + 1) *
  // thread_lookup_size)
  uint32_t h_array_size = size_hash_table / num_hash_func;
  size_t num_threads = blockDim.x * gridDim.x;
  size_t thread_lookup_size = num_array_key / num_threads + 1;

  // go over the array of keys assigned to thread `i`
  size_t array_key_start = i * thread_lookup_size;
  size_t array_key_end = (i + 1) * thread_lookup_size;
  array_key_end = array_key_end > num_array_key ? num_array_key : array_key_end;
  for (size_t k = array_key_start; k < array_key_end; ++k) {
    // for each array corresponding to a hash function
    for (size_t j = 0; j < num_hash_func; ++j) {
      uint32_t pos = hash_device(
          j, array_key[k], &array_coef[j * NUM_HASH_FUNC_COEF], h_array_size);
      if (ptr_hash_table[pos] == array_key[k]) {
        ptr_hash_table[pos] = 0;
      }
    }
  }
}

// insert
// res_pos_hash_func, res_pos_hash_array are used for lookup
__global__ void insert_global(uint32_t *ptr_hash_table, size_t size_hash_table,
                              size_t num_hash_func, uint64_t *array_coef,
                              uint32_t max_eviction, uint32_t *array_key,
                              uint32_t num_array_key,
                              uint32_t *exceed_max_eviction,
                              uint32_t if_replace_key) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // insert the key in the range [i * thread_lookup_size, (i + 1) *
  // thread_lookup_size)
  uint32_t h_array_size = size_hash_table / num_hash_func;
  size_t num_threads = blockDim.x * gridDim.x;
  size_t thread_lookup_size = num_array_key / num_threads + 1;

  // go over the array of keys assigned to thread `i`
  size_t array_key_start = i * thread_lookup_size;
  size_t array_key_end = (i + 1) * thread_lookup_size;
  array_key_end = array_key_end > num_array_key ? num_array_key : array_key_end;
  for (size_t k = array_key_start; k < array_key_end; ++k) {
    uint32_t hash_value;
    bool if_found = false;
    for (size_t j = 0; j < num_hash_func; ++j) {
      hash_value = hash_device(
          j, array_key[k], &array_coef[j * NUM_HASH_FUNC_COEF], h_array_size);
      if (ptr_hash_table[hash_value] == array_key[k]) {
        if (if_replace_key) {
          array_key[k] = 0;
        }
        if_found = true;
        break;
      }
    }
    if (if_found) {
      continue;
    }

    size_t j = 0;
    uint32_t key = array_key[k];
    for (; j < max_eviction; j++) {
      size_t m = j % num_hash_func;
      // calc hash value
      hash_value = hash_device(m, key, &array_coef[m * NUM_HASH_FUNC_COEF],
                               h_array_size);
      // try to insert
      // if failed, exchange the key with the existing key
      uint32_t old_value = atomicExch(&ptr_hash_table[hash_value], key);
      if (old_value == 0 || old_value == key) {
        if (if_replace_key) {
          array_key[k] = 0;
        }
        break;
      } else {
        key = old_value;
      }
    }
    // if exceed the max eviction, set exceed_max_eviction to true
    if (j >= max_eviction) {
      *exceed_max_eviction = 1;
    }
  }
}

// device functions
__device__ void lookup_device(uint32_t *hash_table, size_t size_hash_table,
                              uint32_t *array_key, size_t num_array_key,
                              uint64_t *array_coef, size_t num_hash_func,
                              uint32_t *res_pos_hash_func,
                              uint32_t *res_pos_hash_array) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  // lookup the key in the range [i * thread_lookup_size, (i + 1) *
  // thread_lookup_size)
  uint32_t h_array_size = size_hash_table / num_hash_func;
  size_t num_threads = blockDim.x * gridDim.x;
  size_t thread_lookup_size = num_array_key / num_threads + 1;

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
      uint32_t pos = hash_device(
          j, array_key[k], &array_coef[j * NUM_HASH_FUNC_COEF], h_array_size);
      if (hash_table[pos] == array_key[k]) {
        res_pos_hash_func[k] = i;
        res_pos_hash_array[k] = pos;
      }
    }
  }
}

void CuckooHashCUDA::rehash() {
  auto s_hash_table = get_size_hash_table();

  uint32_t *ptr_cuda_new_hash_table = nullptr;
  cudaMallocManaged(&ptr_cuda_new_hash_table, s_hash_table * sizeof(uint32_t));
  auto new_hash_table = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_new_hash_table, &cudaMemDeconstructor);
  std::swap(hash_table, new_hash_table);

  size_t table_element_cnt = 0;
  for (size_t i = 0; i < s_hash_table; ++i) {
    if (new_hash_table[i] != 0) {
      new_hash_table[table_element_cnt] = new_hash_table[i];
      table_element_cnt++;
    }
  }
  std::cout << "Rehashing " << table_element_cnt << " elements" << std::endl;

  // insert
  size_t num_threads = block_size;
  size_t num_blocks = grid_size;
  uint32_t *ptr_cuda_exceed_max_eviction = nullptr;
  cudaMallocManaged(&ptr_cuda_exceed_max_eviction, sizeof(uint32_t));
  auto exceed_max_eviction = std::unique_ptr<uint32_t, cudaMemDeconstructor_t>(
      ptr_cuda_exceed_max_eviction, &cudaMemDeconstructor);

  do {
    std::cout << "Rehashing... " << std::endl;
    set_max_eviction(get_max_eviction() + eviction_chain_increment);
    generate_hash_func_coef();
    cudaMemset(hash_table.get(), 0, s_hash_table * sizeof(uint32_t));
    *exceed_max_eviction = 0;

    insert_global<<<num_blocks, num_threads>>>(
        hash_table.get(), s_hash_table, get_num_hash_func(),
        hash_func_coef.get(), get_max_eviction(), new_hash_table.get(),
        table_element_cnt, exceed_max_eviction.get(), 0);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    }
  } while (*exceed_max_eviction);
}

hashTablePos CuckooHashCUDA::lookup(uint32_t) {
  std::cout << "Do not call lookup(uint32_t key) in CuckooHashCUDA\n";
  return {UINT_MAX, UINT_MAX};
}
std::vector<hashTablePos> CuckooHashCUDA::lookup(Instruction inst) {
  if (inst.first != "lookup") {
    return {};
  }

#ifdef BENCHMARK
  uint64_t t_start = std::clock();
#endif

  size_t num_array_key = inst.second.size();
  uint32_t *ptr_cuda_array_key = nullptr;
  cudaMallocManaged(&ptr_cuda_array_key, num_array_key * sizeof(uint32_t));
  cudaMemset(ptr_cuda_array_key, 0, num_array_key * sizeof(uint32_t));
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
  cudaMemset(ptr_cuda_res_pos_hash_func, 0, num_array_key * sizeof(uint32_t));
  cudaMallocManaged(&ptr_cuda_res_pos_hash_array,
                    num_array_key * sizeof(uint32_t));
  cudaMemset(ptr_cuda_res_pos_hash_array, 0, num_array_key * sizeof(uint32_t));
  auto res_pos_hash_func = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_res_pos_hash_func, &cudaMemDeconstructor);
  auto res_pos_hash_array = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_res_pos_hash_array, &cudaMemDeconstructor);

  lookup_global<<<num_blocks, num_threads>>>(
      hash_table.get(), get_size_hash_table(), get_num_hash_func(),
      hash_func_coef.get(), array_key.get(), num_array_key,
      res_pos_hash_func.get(), res_pos_hash_array.get());
  cudaDeviceSynchronize();

  std::vector<hashTablePos> results;
#pragma omp parallel for
  for (size_t i = 0; i < num_array_key; ++i) {
    results.push_back({res_pos_hash_func[i], res_pos_hash_array[i]});
  }

#ifdef BENCHMARK
  uint64_t t_end = std::clock();
  std::cout << "Time elapsed in looking up " << inst.second.size()
            << " items: " << (double)(t_end - t_start) / CLOCKS_PER_SEC << "\n";
#endif

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

#ifdef BENCHMARK
  uint64_t t_start = std::clock();
#endif

  size_t max_eviction = 4 * std::log2(inst.second.size());
  max_eviction = max_eviction < 4 ? 4 : max_eviction;
  set_max_eviction(max_eviction);
  std::cout << "Set max eviction: " << get_max_eviction() << std::endl;

  size_t num_array_key = inst.second.size();
  uint32_t *ptr_cuda_array_key = nullptr;
  cudaMallocManaged(&ptr_cuda_array_key, num_array_key * sizeof(uint32_t));
  cudaMemset(ptr_cuda_array_key, 0, num_array_key * sizeof(uint32_t));
  auto array_key = std::unique_ptr<uint32_t[], cudaMemDeconstructor_t>(
      ptr_cuda_array_key, &cudaMemDeconstructor);
#pragma omp parallel for
  for (size_t i = 0; i < num_array_key; ++i) {
    array_key[i] = inst.second[i];
  }

  size_t num_threads = block_size;
  size_t num_blocks = grid_size;
  uint32_t *ptr_cuda_exceed_max_eviction = nullptr;
  cudaMallocManaged(&ptr_cuda_exceed_max_eviction, sizeof(uint32_t));
  auto exceed_max_eviction = std::unique_ptr<uint32_t, cudaMemDeconstructor_t>(
      ptr_cuda_exceed_max_eviction, &cudaMemDeconstructor);

  while (1) {
    *exceed_max_eviction = 0;
    insert_global<<<num_blocks, num_threads>>>(
        hash_table.get(), get_size_hash_table(), get_num_hash_func(),
        hash_func_coef.get(), max_eviction, array_key.get(), num_array_key,
        exceed_max_eviction.get(), 1);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    if (*exceed_max_eviction) {
      int cnt = 0;
      for (size_t i = 0; i < num_array_key; ++i) {
        if (array_key[i] != 0) {
          array_key[cnt] = array_key[i];
          cnt++;
        }
      }
      num_array_key = cnt;
      std::cout << cnt << " keys left to insert\n";

      rehash();
    } else {
      break;
    }
  }

#ifdef BENCHMARK
  uint64_t t_end = std::clock();
  std::cout << "Time elapsed in inserting " << inst.second.size()
            << " items: " << (double)(t_end - t_start) / CLOCKS_PER_SEC << "\n";
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