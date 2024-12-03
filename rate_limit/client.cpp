#include <cassert>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include "asio.hpp"
#include "asio/buffer.hpp"
#include "asio/ip/tcp.hpp"
#include "asio/write.hpp"
#include "utils.h"

using asio::ip::tcp;

class rate_limiter {
public:
  rate_limiter(std::size_t rate) : rate_(rate) {}
  ~rate_limiter() = default;

  void send_data(char *data, std::size_t len, tcp::socket &destination) {
    std::size_t beginning = 0;
    std::size_t left_len = len;
    while (beginning < len) {
      std::size_t leak_data_size = leak();
      if (leak_data_size > 0) {
        asio::write(destination,
                    asio::buffer(data + beginning, leak_data_size));
        beginning += leak_data_size;
      }

      std::size_t len_to_addin =
          std::min(left_len, bucket_capacity_ - used_size_);
      if (len_to_addin == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      } else {
        used_size_ += len_to_addin;
        left_len -= len_to_addin;
      }
    }
    assert(used_size_ == 0);
    assert(left_len == 0);
    assert(beginning == len);
  }

private:
  std::size_t leak() {
    auto now = std::chrono::steady_clock::now();
    double time_passed =
        std::chrono::duration<double>(now - last_leak_time_).count();

    std::size_t leak_data_size = static_cast<std::size_t>(rate_ * time_passed);
    if (leak_data_size > used_size_) {
      leak_data_size = used_size_;
    }

    used_size_ -= leak_data_size;

    last_leak_time_ = now;

    return leak_data_size;
  }

  double rate_;
  std::size_t bucket_capacity_{10 * 1024 * 1024}; // 10MB字节
  std::size_t used_size_{0};

  std::chrono::steady_clock::time_point last_leak_time_{
      std::chrono::steady_clock::now()};
};

int main() {
  asio::io_context io_context;
  tcp::socket destination(io_context);
  tcp::resolver resolver(io_context);
  asio::connect(destination, resolver.resolve("0.0.0.0", "11223"));

  std::vector<char> data(data_transfer_size);
  rate_limiter rl{rate_limit};
  rl.send_data(data.data(), data.size(), destination);

  return 0;
}