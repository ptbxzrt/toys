#include "ylt/coro_rpc/impl/coro_rpc_client.hpp"
#include "ylt/coro_rpc/impl/default_config/coro_rpc_config.hpp"
#include <memory>

class proxy {
public:
  proxy();

  void start();

  void func2();

private:
  std::unique_ptr<coro_rpc::coro_rpc_server> rpc_server_{nullptr};
  std::unique_ptr<coro_rpc::coro_rpc_client> coordinator_{nullptr};
};