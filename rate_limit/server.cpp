#include "asio/buffer.hpp"
#include "asio/ip/tcp.hpp"
#include "asio/read.hpp"
#include "utils.h"
#include <fmt/core.h>
#include <iostream>

using asio::ip::tcp;

int main() {
  asio::io_context io_context;

  tcp::socket server =
      tcp::acceptor(io_context, tcp::endpoint(tcp::v4(), 11223)).accept();

  std::vector<char> data(data_transfer_size);
  auto now = std::chrono::steady_clock::now();

  asio::read(server, asio::buffer(data.data(), data.size()));

  auto after = std::chrono::steady_clock::now();
  double time_passed = std::chrono::duration<double>(after - now).count();
  double rate =
      static_cast<double>(data_transfer_size) / time_passed / 1024 / 1024;
  std::cout << fmt::format("传输{}MB的数据耗时{}秒，传输速率为{}MB/s\n",
                           data_transfer_size / 1024 / 1024, time_passed, rate);

  return 0;
}