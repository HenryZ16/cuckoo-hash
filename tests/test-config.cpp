#include <format>
#include <iostream>

#include "config.h"
#include "utils.h"

const std::string INPUT_FILE = "config.txt";

int main() {
  std::ios::sync_with_stdio(false);
  printlog("Start testing");

  cuckooHash::Config config(INPUT_FILE);
  printlog("num_hash_func: {}", config.get_num_hash_func());
  printlog("size_hash_table: {}", config.get_size_hash_table());
  printlog("input_file: {}", config.get_input_file());

  return 0;
}