#pragma once

#include <fstream>
#include <string>

namespace cuckooHash {

/* configs:
 *   # The number of hash functions used in Cuckoo Hash.
 *   num_hash_func   <size_t>
 *
 *   # The size of hash table. The unit can be 'KB', 'MB', 'GB'.
 *   size_hash_table <size_t> [unit]
 *
 *   # The input file used to benchmark. If <filename> is stdin, then receive
 * commands from console.
 *   input_file      <string>
 *
 *   # mark if the input file is binary. can be `0` or `1`
 *   is_binary       <int>
 *
 *   # The increment of the eviction chain.
 *   eviction_chain_increment <size_t>
 */

class Config {
  size_t num_hash_func;
  size_t size_hash_table;
  std::string input_file;
  std::string dump_file;
  bool is_binary;
  size_t eviction_chain_increment;

public:
  // Constructors
  Config() = delete;
  Config(const size_t num_hash_func, const size_t size_hash_table,
         const std::string &input_file, const std::string &dump_file,
         const bool is_binary = false,
         const size_t eviction_chain_increment = 4)
      : num_hash_func(num_hash_func), size_hash_table(size_hash_table),
        input_file(input_file), dump_file(dump_file), is_binary(is_binary),
        eviction_chain_increment(eviction_chain_increment) {}
  explicit Config(const std::string &filename);

  // Default destructor, copy constructor, and copy assignment operator
  // member Accessors
  size_t get_num_hash_func() const { return num_hash_func; }
  size_t get_size_hash_table() const { return size_hash_table; }
  std::string get_input_file() const { return input_file; }
  std::string get_dump_file() const { return dump_file; }
  bool get_is_binary() const { return is_binary; }
  size_t get_eviction_chain_increment() const {
    return eviction_chain_increment;
  }
};
} // namespace cuckooHash