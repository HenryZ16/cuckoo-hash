#include <cmath>
#include <ctime>
#include <format>
#include <iostream>
#include <vector>

#include "cuckooHash.h"
#include "cuckooHashDefault.h"
#include "input.h"
#include "utils.h"

namespace cuckooHash {
void CuckooHashDefault::rehash() {
  auto n_hash_func = get_num_hash_func();
  auto s_hash_table = get_size_hash_table();
  generate_hash_func_coef();

  auto new_hash_table = std::vector<std::unique_ptr<int[]>>(n_hash_func);
  for (auto &uptr_table : new_hash_table) {
    uptr_table = std::make_unique<int[]>(s_hash_table);
  }
  for (size_t i = 0; i < n_hash_func; ++i) {
    for (size_t j = 0; j < s_hash_table; ++j) {
      std::swap(new_hash_table[i][j], hash_table[i][j]);
    }
  }

  for (auto &uptr_table : new_hash_table) {
    for (size_t i = 0; i < s_hash_table; ++i) {
      insert(uptr_table[i]);
    }
  }
}

hashTablePos CuckooHashDefault::lookup(int key) {
  size_t n_hash_func = get_num_hash_func();
  std::vector<int> hash_values(n_hash_func);
  for (size_t i = 0; i < n_hash_func; ++i) {
    hash_values[i] = hash(key, i);
  }

  hashTablePos success = {0, 0};
  for (auto &table : hash_table) {
    for (size_t i = 0; i < n_hash_func; ++i) {
      if (table[hash_values[i]] == key) {
        success = {i, hash_values[i]};
        break;
      }
    }
  }

  return success;
}

void CuckooHashDefault::delete_key(int key) {
  hashTablePos pos = lookup(key);
  if (pos.first == 0 && pos.second == 0) {
    return;
  }

  hash_table[pos.first][pos.second] = 0;
}

void CuckooHashDefault::insert(int key) {
  size_t n_hash_func = get_num_hash_func();
  size_t max_eviction = get_max_eviction();
  std::vector<int> hash_values(n_hash_func);

  for (size_t i = 0; i < max_eviction; i += n_hash_func) {
    for (size_t j = 0; j < n_hash_func; ++j) {
      hash_values[j] = hash(key, j);
      if (hash_table[j][hash_values[j]] == 0) {
        hash_table[j][hash_values[j]] = key;
        return;
      }
      std::swap(key, hash_table[j][hash_values[j]]);
    }
  }

  rehash();
  insert(key);
}

void CuckooHashDefault::insert(Instruction inst) {
  if (inst.first != "insert") {
    return;
  }
  set_max_eviction(4 * std::log(inst.second.size()));
#ifdef BENCHMARK
  uint64_t t_start = std::clock();
#endif
  for (const auto &key : inst.second) {
    insert(key);
  }
#ifdef BENCHMARK
  uint64_t t_end = std::clock();
  println("Time elapsed in inserting {} items: {}", inst.second.size(),
          t_end - t_start);
#endif
}

void CuckooHashDefault::print() { println("Implement print()"); }
} // namespace cuckooHash