#include <cmath>
#include <ctime>
#include <vector>

#include "cuckooHash.h"
#include "cuckooHashDefault.h"
#include "input.h"
#include "utils.h"

namespace cuckooHash {
void CuckooHashDefault::rehash() {
  println("Rehashing...");
  auto s_hash_table = get_size_hash_table();
  generate_hash_func_coef();
#ifdef DEBUG
  println("Hash table before rehash:");
  print();
  println("Rehash with coef: [{}, {}], [{}, {}]", hash_func_coef[0].first,
          hash_func_coef[0].second, hash_func_coef[1].first,
          hash_func_coef[1].second);
#endif

  auto new_hash_table = std::unique_ptr<uint32_t[]>(new uint32_t[s_hash_table]);
  std::swap(hash_table, new_hash_table);

  for (size_t i = 0; i < s_hash_table; ++i) {
    if (new_hash_table[i] != 0) {
      insert(new_hash_table[i]);
    }
  }
#ifdef DEBUG
  println("Hash table after rehash:");
  print();
#endif
}

hashTablePos CuckooHashDefault::lookup(uint32_t key) {
  size_t n_hash_func = get_num_hash_func();
  std::vector<uint32_t> hash_values(n_hash_func);
  for (size_t i = 0; i < n_hash_func; ++i) {
    hash_values[i] = hash(key, i);
  }

  hashTablePos success = {0, 0};
  for (size_t i = 0; i < n_hash_func; ++i) {
    if (hash_table[hash_values[i]] == key) {
      success = {i, hash_values[i]};
      break;
    }
  }

  return success;
}

void CuckooHashDefault::delete_key(uint32_t key) {
  hashTablePos pos = lookup(key);
  if (pos == std::make_pair(0u, 0u)) {
    return;
  }

  hash_table[pos.second] = 0;
}

void CuckooHashDefault::insert(uint32_t key) {
  hashTablePos pos = lookup(key);
  if (key == 0 || pos != std::make_pair(0u, 0u)) {
    return;
  }

  size_t n_hash_func = get_num_hash_func();
  size_t max_eviction = get_max_eviction();
  uint32_t hash_value;

  for (size_t i = 0; i < max_eviction; i += n_hash_func) {
    for (size_t j = 0; j < n_hash_func; ++j) {
      hash_value = hash(key, j);
      if (hash_table[hash_value] == 0) {
        hash_table[hash_value] = key;
        return;
      }
      std::swap(key, hash_table[hash_value]);
    }
  }

  rehash();
  insert(key);
}

void CuckooHashDefault::insert(Instruction inst) {
  if (inst.first != "insert") {
    return;
  }
  size_t max_eviction = 4 * std::log2(inst.second.size());
  max_eviction = max_eviction < 4 ? 4 : max_eviction;
  set_max_eviction(max_eviction);
  println("Set max eviction: {}", get_max_eviction());
#ifdef BENCHMARK
  uint64_t t_start = std::clock();
#endif
  for (size_t i = 0; i < inst.second.size(); ++i) {
    if (i % 100000 == 0 || inst.second[i] == 0) {
      println("Inserting {}-th item: {}", i, inst.second[i]);
    }
    insert(inst.second[i]);
  }
#ifdef BENCHMARK
  uint64_t t_end = std::clock();
  println("Time elapsed in inserting {} items: {}", inst.second.size(),
          t_end - t_start);
#endif
#ifdef DEBUG
  println("Final hash table:");
  print();
#endif
}

void CuckooHashDefault::print() {
  size_t s_hash_table = get_size_hash_table();
  println("Table: ");
  for (size_t j = 0; j < s_hash_table; j++) {
    std::cout << hash_table[j] << " ";
    if (j % 8 == 0) {
      println("");
    }
  }
  println("");
}
} // namespace cuckooHash