#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <fmt/core.h>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class producer_consumer {
public:
  void produce(int id, int data) {
    while (!flag) {
      {
        std::unique_lock<std::mutex> lck(m);
        sources.push(data);
      }
      cv.notify_one();
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << fmt::format("生产者{}生产了{}\n", id, data);
    }
  }

  void consume(int id) {
    while (true) {
      int data{};
      {
        std::unique_lock<std::mutex> lck(m);
        cv.wait(lck, [this]() -> bool { return flag || !sources.empty(); });
        if (flag) {
          return;
        }
        data = sources.front();
        sources.pop();
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << fmt::format("消费者{}消费了{}\n", id, data);
    }
  }

  void stop() {
    flag = true;
    cv.notify_all();
  }

private:
  std::mutex m{};
  std::condition_variable cv{};

  bool flag{false};
  std::queue<int> sources{};
};

int main() {
  constexpr std::size_t producer_num = 5;
  constexpr std::size_t consumer_num = 10;
  producer_consumer pc{};
  std::vector<std::jthread> producers{};
  std::vector<std::jthread> consumers{};
  for (std::size_t i = 0; i < producer_num; i++) {
    std::jthread producer(&producer_consumer::produce, &pc, i, std::rand());
    producers.push_back(std::move(producer));
  }
  for (std::size_t i = 0; i < consumer_num; i++) {
    std::jthread consumer(&producer_consumer::consume, &pc, i);
    consumers.push_back(std::move(consumer));
  }
  std::this_thread::sleep_for(std::chrono::seconds(5));
  pc.stop();
  return 0;
}