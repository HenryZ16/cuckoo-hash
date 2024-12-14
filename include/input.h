#pragma once

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

public:
  // Constructors
  Input() : file(&std::cin), file_stream() {}
  explicit Input(const std::string &filename)
      : file(&file_stream), file_stream(filename) {}

  // Default destructor, copy constructor, and copy assignment operator
  // member Accessors

  // returns {"stop", {0}} if there is no more instruction
  Instruction fetch_instruction();
};
} // namespace cuckooHash