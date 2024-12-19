#include <cstdint>
#include <fstream>
#include <memory>
#include <random>

#include "utils.h"

// generate insert instruction for 16777216 keys, then generate lookup
// instruction 11 times for the keys. For the i-th lookup, 100 - 10i keys are
// randomly chosen from the 16777216 keys, and the rest are random keys.
int gen_exp2(const std::string &output_file);

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
      output_file.close();
      return 1;
    }
    if (instruction == "exp2") {
      auto ret = gen_exp2(argv[1]);
      output_file.close();
      return ret;
    }
    if (instruction != "insert" && instruction != "delete" &&
        instruction != "lookup") {
      println("Invalid instruction: {}", instruction);
      output_file.close();
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

int gen_exp2(const std::string &output_file) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(1, UINT32_MAX);
  std::ofstream output_file_stream(output_file + "_insert", std::ios::binary);

  uint32_t instruction_code = 1;
  uint32_t cnt = 16777216;
  auto S = std::unique_ptr<uint32_t[]>(new uint32_t[cnt]);
  uint32_t header = (instruction_code << 30) | cnt;
  output_file_stream.write(reinterpret_cast<char *>(&header), sizeof(header));
  for (uint32_t i = 0; i < cnt; ++i) {
    uint32_t key = dis(gen);
    S[i] = key;
    output_file_stream.write(reinterpret_cast<char *>(&key), sizeof(key));
  }
  std::cout << "Generated 16777216 keys\n";

  std::uniform_int_distribution<uint32_t> dis_S(0, cnt - 1);
  for (int i = 0; i < 11; ++i) {
    std::ofstream output_file_stream(
        output_file + "_lookup_" + std::to_string(i), std::ios::binary);
    auto S_cnt = (100 - 10 * i) * cnt / 100;
    header = (2 << 30) | cnt;
    output_file_stream.write(reinterpret_cast<char *>(&header), sizeof(header));
    for (uint32_t i = 0; i < S_cnt; ++i) {
      uint32_t key = S[dis_S(gen)];
      output_file_stream.write(reinterpret_cast<char *>(&key), sizeof(key));
    }
    for (uint32_t i = 0; i < cnt - S_cnt; ++i) {
      uint32_t key = dis(gen);
      output_file_stream.write(reinterpret_cast<char *>(&key), sizeof(key));
    }
    std::cout << "Generated " << i << "-th lookup\n";
  }

  return 0;
}