#include "coordinator.h"
#include "proxy.h"
#include <chrono>
#include <future>
#include <latch>
#include <thread>
#include <utility>

coordinator::coordinator() {
  rpc_server_ = std::make_unique<coro_rpc::coro_rpc_server>(3, 11111);
  rpc_server_->register_handler<&coordinator::func1>(this);
  rpc_server_->register_handler<&coordinator::func3>(this);

  proxy_ = std::make_unique<coro_rpc::coro_rpc_client>();
  async_simple::coro::syncAwait(proxy_->connect("0.0.0.0", "21111"));

  std::latch sync{2};
  worker_ = std::thread([this, &sync]() {
    auto &&f = rpc_server_->async_start();
    sync.count_down();
    std::move(f).get();
  });

  sync.arrive_and_wait();
  func1();
}

void coordinator::start() { worker_.join(); }

void coordinator::func1() {
  std::cout << fmt::format("func1: hello coordinator\n");

  async_simple::coro::syncAwait(proxy_->call<&proxy::func2>());
}

void coordinator::func3() {
  std::cout << fmt::format("func3: hello coordinator\n");
}

int main() {
  coordinator c;
  c.start();
  return 0;
}