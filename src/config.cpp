#include "config.h"
#include <fstream>
#include <string>

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

namespace cuckooHash {
Config::Config(const std::string &filename)
    : num_hash_func(0), size_hash_table(0), input_file(filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    size_t pos = line.find(' ');
    if (pos == std::string::npos) {
      throw std::runtime_error("Invalid config file format");
    }

    std::string key = line.substr(0, pos);
    std::string value = line.substr(pos + 1);

    size_t pos_hash = value.find('#');
    if (pos_hash != std::string::npos) {
      value = value.substr(0, pos_hash);
    }
    value.erase(0, value.find_first_not_of(" \n\r\t"));
    value.erase(value.find_last_not_of(" \n\r\t") + 1);

    if (key == "num_hash_func") {
      num_hash_func = std::stoul(value);
    } else if (key == "size_hash_table") {
      pos = value.find(' ');
      if (pos != std::string::npos) {
        std::string unit = value.substr(pos + 1);
        value = value.substr(0, pos);
        if (unit == "KB") {
          size_hash_table = std::stoul(value) * 1024;
        } else if (unit == "MB") {
          size_hash_table = std::stoul(value) * 1024 * 1024;
        } else if (unit == "GB") {
          size_hash_table = std::stoul(value) * 1024 * 1024 * 1024;
        } else {
          throw std::runtime_error("Invalid unit: " + unit);
        }
      } else {
        size_hash_table = std::stoul(value);
      }
    } else if (key == "input_file") {
      input_file = value;
    } else {
      throw std::runtime_error("Invalid config key: " + key);
    }
  }
}
} // namespace cuckooHash