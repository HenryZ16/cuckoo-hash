#pragma once

#include "config.h"
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace cuckooHash {
typedef std::pair<std::string, std::vector<uint32_t>> Instruction;

class Input {
  /*  Cuckoo Hash follows the rules below to process the input:
      ```
      <instruction> <cnt>
        <id>
        ...
        <id>
      <instruction> <cnt>
      ...
      ```
      where:
        - `<instruction>` can be `lookup` `insert`, `delete`.
        - `<cnt>` is the number of `<id>`s, which is <= 2^31 - 1.
        - `<id>` is the key, which is <= 2^31 - 1.
   */
  std::istream *file;
  std::ifstream file_stream;
  bool is_binary;

public:
  // Constructors
  Input() : file(&std::cin), file_stream() {}
  explicit Input(const std::string &filename, const bool is_binary = false)
      : file(&file_stream), file_stream(filename), is_binary(is_binary) {}
  explicit Input(const Config config)
      : Input(config.get_input_file(), config.get_is_binary()) {}

  // Default destructor, copy constructor, and copy assignment operator
  // member Accessors

  // returns {"stop", {0}} if there is no more instruction
  Instruction fetch_instruction();
  Instruction fetch_instruction_text();
  Instruction fetch_instruction_binary();
};
} // namespace cuckooHash