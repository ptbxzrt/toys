#include "ylt/coro_rpc/impl/coro_rpc_client.hpp"
#include "ylt/coro_rpc/impl/default_config/coro_rpc_config.hpp"
#include <memory>
#include <thread>

class coordinator {
public:
  coordinator();

  void start();

  void func1();

  void func3();

private:
  std::unique_ptr<coro_rpc::coro_rpc_server> rpc_server_{nullptr};
  std::unique_ptr<coro_rpc::coro_rpc_client> proxy_{nullptr};

  std::thread worker_{};
};