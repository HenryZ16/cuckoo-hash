#include <cstdint>
#include <fstream>
#include <random>

#include "utils.h"

// Generate a file with the format:
// <instruction:2> <cnt:30> <key:32> ... <key:32>
// receive the argument as: <output_file> <instruction> <cnt>
int main(int argc, char **argv) {
  if (argc < 4) {
    println(
        "Usage: {} <output_file> <instruction> <cnt> [<instruction> <cnt>] ...",
        argv[0]);
    return -1;
  }
  std::ofstream output_file(argv[1], std::ios::binary);

  for (int i = 2; i < argc; i += 2) {
    std::string instruction = argv[i];
    uint32_t cnt = std::stoi(argv[i + 1]);
    if (!output_file.is_open()) {
      println("Failed to open file: {}", argv[1]);
      return 1;
    }
    if (instruction != "insert" && instruction != "delete" &&
        instruction != "lookup") {
      println("Invalid instruction: {}", instruction);
      return 1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(1, UINT32_MAX);

    uint32_t instruction_code = 1;
    if (instruction == "lookup") {
      instruction_code = 2;
    } else if (instruction == "delete") {
      instruction_code = 3;
    }
    uint32_t header = (instruction_code << 30) | cnt;
    output_file.write(reinterpret_cast<char *>(&header), sizeof(header));
    for (uint32_t i = 0; i < cnt; ++i) {
      uint32_t key = dis(gen);
      output_file.write(reinterpret_cast<char *>(&key), sizeof(key));
    }
  }

  output_file.close();
  return 0;
}