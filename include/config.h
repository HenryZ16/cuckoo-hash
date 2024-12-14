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
 */

class Config {
  size_t num_hash_func;
  size_t size_hash_table;
  std::string input_file;

public:
  // Constructors
  Config() = delete;
  Config(const size_t num_hash_func, const size_t size_hash_table,
         const std::string &input_file)
      : num_hash_func(num_hash_func), size_hash_table(size_hash_table),
        input_file(input_file) {}
  explicit Config(const std::string &filename);

  // Default destructor, copy constructor, and copy assignment operator
  // member Accessors
  size_t get_num_hash_func() const { return num_hash_func; }
  size_t get_size_hash_table() const { return size_hash_table; }
  std::string get_input_file() const { return input_file; }
};
} // namespace cuckooHash