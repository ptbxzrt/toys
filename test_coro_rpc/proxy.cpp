#include "proxy.h"
#include "coordinator.h"

proxy::proxy() {
  rpc_server_ = std::make_unique<coro_rpc::coro_rpc_server>(3, 21111);

  rpc_server_->register_handler<&proxy::func2>(this);
}

void proxy::start() { rpc_server_->start(); }

void proxy::func2() {
  std::cout << std::format("func2: hello proxy\n");

  coordinator_ = std::make_unique<coro_rpc::coro_rpc_client>();
  async_simple::coro::syncAwait(coordinator_->connect("0.0.0.0", "11111"));
  async_simple::coro::syncAwait(coordinator_->call<&coordinator::func3>());
}

int main() {
  proxy p;
  p.start();
  return 0;
}