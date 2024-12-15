#include "input.h"
#include "utils.h"

namespace cuckooHash {
Instruction Input::fetch_instruction() {
  return is_binary ? fetch_instruction_binary() : fetch_instruction_text();
}

Instruction Input::fetch_instruction_text() {
  std::string instruction;
  uint32_t cnt;
  std::vector<uint32_t> ids;

  // read header
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

Instruction Input::fetch_instruction_binary() {
  uint32_t header;
  if (!file_stream.read(reinterpret_cast<char *>(&header), sizeof(header))) {
    return {"stop", {0}};
  }

  std::string instruction;
  uint32_t cnt;
  uint8_t instruction_code = header >> 30;
  if (instruction_code == 1) {
    instruction = "insert";
  } else if (instruction_code == 2) {
    instruction = "lookup";
  } else if (instruction_code == 3) {
    instruction = "delete";
  }
  cnt = header & 0x3FFFFFFF;

  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < cnt; i++) {
    uint32_t id;
    if (!file_stream.read(reinterpret_cast<char *>(&id), sizeof(id))) {
      cnt = i;
      break;
    }
    ids.push_back(id);
  }

  return {instruction, ids};
}
} // namespace cuckooHash