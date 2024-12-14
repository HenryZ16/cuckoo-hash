#include <format>
#include <iostream>
#include <string>

#include "input.h"
#include "utils.h"

const std::string INPUT_FILE = "input.txt";

int main() {
  std::ios::sync_with_stdio(false);
  printlog("Start testing");

  cuckooHash::Input input(INPUT_FILE);
  while (true) {
    auto [instruction, ids] = input.fetch_instruction();
    if (instruction == "stop") {
      break;
    }

    println("Instruction: {}", instruction);
    for (const auto &id : ids) {
      println("  id: {}", id);
    }
  }

  return 0;
}