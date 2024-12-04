#include "../tools.hpp"
#include <cassert>
#include <cmath>
#include <format>
#include <iostream>

enum class replication_schema { random, copyset };

struct config {
  double failure_rate;
  int num_nodes;
  int replication_factor;
  int blocks_per_node;
  int scatter_width;
};

template <replication_schema SCHEMA> class loss_probability_replication {
public:
  loss_probability_replication(const config &conf) : conf_(conf) {}
  ~loss_probability_replication() = default;

  double by_equation() {
    int num_failed_nodes = conf_.num_nodes * conf_.failure_rate;
    if (num_failed_nodes < conf_.replication_factor) {
      return 0;
    }

    double not_lose_one_unique_block_probability =
        1.0 - get_lose_one_unique_block_probability();

    int num_unique_blocks =
        conf_.blocks_per_node * conf_.num_nodes / conf_.replication_factor;

    return 1.0 -
           std::pow(not_lose_one_unique_block_probability, num_unique_blocks);
  }

private:
  double get_lose_one_unique_block_probability() {
    int num_failed_nodes = conf_.num_nodes * conf_.failure_rate;
    double failure_case =
        get_combination(num_failed_nodes, conf_.replication_factor);
    double total_case =
        get_combination(conf_.num_nodes, conf_.replication_factor);
    if constexpr (SCHEMA == replication_schema::random) {
      return failure_case / total_case;
    } else if (SCHEMA == replication_schema::copyset) {
      assert(conf_.scatter_width > 0);
      int num_copyset_per_permutation =
          conf_.num_nodes / conf_.replication_factor;
      int num_permutation =
          conf_.scatter_width / (conf_.replication_factor - 1);
      num_permutation +=
          (conf_.scatter_width % (conf_.replication_factor - 1) == 0 ? 0 : 1);
      int num_copyset = num_permutation * num_copyset_per_permutation;

      return (failure_case / total_case) * (num_copyset / total_case);
    } else {
      assert(false);
    }
  }

  config conf_;
};

int main() {
  for (int num_nodes = 100; num_nodes <= 10000; num_nodes += 100) {
    double failure_rate = 0.01;
    loss_probability_replication<replication_schema::random> lpr_random(
        {.failure_rate = failure_rate,
         .num_nodes = num_nodes,
         .replication_factor = 3,
         .blocks_per_node = 10000,
         .scatter_width = -1});
    loss_probability_replication<replication_schema::copyset> lpr_copyset(
        {.failure_rate = failure_rate,
         .num_nodes = num_nodes,
         .replication_factor = 3,
         .blocks_per_node = 10000,
         .scatter_width = 200});
    std::cout << std::format(
        "系统共有{:6}个节点，有{:.2f}"
        "的节点损坏，通过公式计算数据丢失概率，随机模式为{}，copyset模式为{}\n",
        num_nodes, failure_rate, lpr_random.by_equation(),
        lpr_copyset.by_equation());
  }

  return 0;
}