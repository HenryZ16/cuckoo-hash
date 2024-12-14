#include <format>
#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout << std::format("[{}] Hellow, World!\n", __FILE__);
  return 0;
}