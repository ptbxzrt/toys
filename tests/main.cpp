#include <format>
#include <iostream>
#include <string>

template <typename T> void func(T &&arg) {
  std::cout << std::format("hello world") << std::endl;
}

int main() {




  func("abc");

  auto s = "abc";
  func(std::string("abc"));
  func("abc");

  int a = 100;
  func(a);
  func(100);

  return 0;
}