#include <string>

#include "config.h"
#include "cuckooHashCUDA.h"
#include "input.h"

// receive the argument as: <config_file>
int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <config_file>";
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

      cuckoo_hash.load(config.get_dump_file());
      if (instruction == "lookup") {
        std::vector<cuckooHash::hashTablePos> t_result;
        std::cout << "Lookup " << ids.size() << " items" << std::endl;
        cuckoo_hash.lookup(instruction_set, t_result);
      } else if (instruction == "insert") {
        std::cout << "Insert " << ids.size() << " items" << std::endl;
        cuckoo_hash.insert(instruction_set);
      } else if (instruction == "delete") {
        std::cout << "Delete " << ids.size() << " items" << std::endl;
        for (auto id : ids) {
          cuckoo_hash.delete_key(id);
        }
      }

      cuckoo_hash.dump(config.get_dump_file());
      std::cout << "Done" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cout << "Error: " << e.what() << std::endl;
    return 1;
  }
}