#include "config.h"
#include "cuckooHashCUDA.h"
#include "input.h"
#include "utils.h"

// receive the argument as: <config_file>
int main(int argc, char **argv) {
  if (argc < 2) {
    println("Usage: {} <config_file>", argv[0]);
    return 1;
  }

  try {
    std::string config_file = argv[1];
    cuckooHash::Config config(config_file);
    cuckooHash::CuckooHashCUDA cuckoo_hash(config);

    cuckooHash::Input input(config);
    while (true) {
      auto instruction_set = input.fetch_instruction();
      auto instruction = instruction_set.first;
      auto ids = instruction_set.second;
      if (instruction == "stop") {
        break;
      }

      if (instruction == "lookup") {
        println("Lookup {} items", ids.size());
        for (auto id : ids) {
          cuckoo_hash.lookup(id);
        }
        println("Done");
      } else if (instruction == "insert") {
        println("Insert {} items", ids.size());
        cuckoo_hash.insert(instruction_set);
        println("Done");
      } else if (instruction == "delete") {
        println("Delete {} items", ids.size());
        for (auto id : ids) {
          cuckoo_hash.delete_key(id);
        }
        println("Done");
      }
    }
  } catch (const std::exception &e) {
    println("Error: {}", e.what());
    return 1;
  }
}