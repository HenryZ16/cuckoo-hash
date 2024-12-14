#include "input.h"

namespace cuckooHash {
Instruction Input::fetch_instruction() {
  std::string instruction;
  uint32_t cnt;
  std::vector<uint32_t> ids;

  if (!(file_stream >> instruction >> cnt)) {
    return {"stop", {0}};
  }
  if (!(instruction == "lookup" || instruction == "insert" ||
        instruction == "delete")) {
    return {"stop", {0}};
  }

  for (uint32_t i = 0; i < cnt; i++) {
    uint32_t id;
    if (!(file_stream >> id)) {
      cnt = i;
      break;
    }
    ids.push_back(id);
  }

  return {instruction, ids};
}
} // namespace cuckooHash