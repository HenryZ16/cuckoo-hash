#pragma once

#include <format>
#include <iostream>

#define println(...)                                                           \
  do {                                                                         \
    std::cout << std::format(__VA_ARGS__) << std::endl;                        \
  } while (0)
#define printlog(...)                                                          \
  do {                                                                         \
    std::cout << std::format("[{}:{}] ", __FILE__, __LINE__)                   \
              << std::format(__VA_ARGS__) << std::endl;                        \
  } while (0)
