#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <format>
#include <ios>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

#include "../tools.hpp"

enum class solution_type { copyset, ecset };

class random_generator {
public:
  random_generator() : rd{}, gen(rd()) {}

  std::mt19937 &get_gen() { return gen; }

private:
  std::random_device rd;
  std::mt19937 gen;
};

struct ec_set_solution_config {
  int expected_scatter_width;

  int K;
  int L;
  int G;

  int num_nodes;
  int num_stripes;

  // 每个块大小（单位MB）
  int block_size;

  void print_config() const {
    std::cout << std::format(
        "EC Set Solution 配置信息: K: {}, L: {}, G: {}, "
        "expected_scatter_width: "
        "{}, num_nodes: {}, num_stripes: {}, block_size: {} MB\n",
        K, L, G, expected_scatter_width, num_nodes, num_stripes, block_size);
  }
};

template <solution_type TYPE> class solution {
public:
  solution(const ec_set_solution_config &config)
      : config_(config), nodes_permutation_(config_.num_nodes),
        repair_load_graph_(config_.num_nodes,
                           std::vector<int>(config_.num_nodes, 0)) {
    assert(config_.K % config_.L == 0);
    len_stripe_ = config_.K + config_.L + config_.G;
    num_data_blocks_each_local_group_ = config_.K / config_.L;

    std::iota(nodes_permutation_.begin(), nodes_permutation_.end(), 0);

    get_num_nodes_permutations();
    // std::cout << num_nodes_permutations_ << std::endl;

    random_generator rd_gen1{}; // 获取节点列表的随机排列
    for (int i = 0; i < num_nodes_permutations_; i++) {
      std::shuffle(nodes_permutation_.begin(), nodes_permutation_.end(),
                   rd_gen1.get_gen());
      for (std::size_t j = 0; j + len_stripe_ < nodes_permutation_.size();
           j += len_stripe_) {
        ec_sets_.push_back(
            std::vector<int>(nodes_permutation_.begin() + j,
                             nodes_permutation_.begin() + j + len_stripe_));
      }
    }

    random_generator rd_gen2{}; // 从诸多ec_set中随机选取一个
    random_generator rd_gen3{}; // shuffle某个ec_set
    std::uniform_int_distribution<std::size_t> dis(0, ec_sets_.size() - 1);
    for (int i = 0; i < config_.num_stripes; i++) {
      auto ec_set = ec_sets_[dis(rd_gen2.get_gen())];
      if constexpr (TYPE == solution_type::ecset) {
        std::shuffle(ec_set.begin(), ec_set.end(), rd_gen3.get_gen());
      }
      for (int block_id = 0; block_id < len_stripe_; block_id++) {
        // repair_group里的元素是node_id
        auto repair_group = get_repair_group(block_id, ec_set);

        int cur_node_id = ec_set[block_id];
        for (auto helper_node_id : repair_group) {
          helpers_of_each_node_[cur_node_id].insert(helper_node_id);
          repair_load_graph_[cur_node_id][helper_node_id]++;
        }
        repair_IO_when_node_fail_[cur_node_id] += repair_group.size();
      }
    }
  }

  void test_scatter_width() {
    double avg_num_helpers = 0;
    double avg_repair_IO = 0;
    double avg_helpers_per_TB = 0;
    for (auto &pr : helpers_of_each_node_) {
      int node_id = pr.first;
      int num_helpers = pr.second.size();
      int repair_IO = repair_IO_when_node_fail_[node_id];
      double helpers_per_TB =
          static_cast<double>(num_helpers) /
          (static_cast<double>(repair_IO) *
           static_cast<double>(config_.block_size) / 1024.0 / 1024.0);

      avg_num_helpers += static_cast<double>(num_helpers);
      avg_repair_IO += static_cast<double>(repair_IO);
      avg_helpers_per_TB += static_cast<double>(helpers_per_TB);

      // std::cout
      //     << std::format(
      //            "{:4}号节点宕机时，有{:4}个节点可以参与修复，会产生{:6}"
      //            "个修复IO（单位为块数量），平均每1TB修复IO有{:4}个节点参与",
      //            node_id, num_helpers, repair_IO, helpers_per_TB)
      //     << std::endl;
    }

    avg_num_helpers /= config_.num_nodes;
    avg_repair_IO /= config_.num_nodes;
    avg_helpers_per_TB /= config_.num_nodes;

    double loss_probability = get_data_loss_probability_when_g_plus_2_fail();

    std::cout
        << std::format(
               "某个节点宕机时，平均有{:.4f}个节点可以参与修复，平均会产生{:."
               "4f}"
               "个修复IO（单位为块数量），平均每1TB修复IO有{:.4f}"
               "个节点参与。当{:2}个节点挂掉时，有{:.8f}概率数据丢失。",
               avg_num_helpers, avg_repair_IO, avg_helpers_per_TB,
               config_.G + 2, loss_probability)
        << std::endl;
  }

  void test_load_graph() {
    double avg = 0;
    double avg_distribution = 0;
    for (int node_id = 0; node_id < config_.num_nodes; node_id++) {
      std::vector<int> repair_IO_distribution{};
      for (int helper_node_id = 0; helper_node_id < config_.num_nodes;
           helper_node_id++) {
        int helper_weight = repair_load_graph_[node_id][helper_node_id];
        if (helper_weight > 0) {
          repair_IO_distribution.push_back(helper_weight);
        }
      }
      auto ret = get_avg_and_variance(repair_IO_distribution);
      avg += ret.first;
      avg_distribution += ret.second;
    }
    avg /= config_.num_nodes;
    avg_distribution /= config_.num_nodes;
    std::cout << std::format("某个节点宕机时，在各个节点上产生的修复IO的平均值"
                             "为{}，variance为{}。",
                             avg, avg_distribution)
              << std::endl;
  }

  const ec_set_solution_config &get_config() { return config_; }

private:
  void get_num_nodes_permutations() {
    if constexpr (TYPE == solution_type::ecset) {
      double blocks_per_node =
          static_cast<double>(config_.num_stripes * (len_stripe_)) /
          static_cast<double>(config_.num_nodes);

      double k = config_.K;
      double l = config_.L;
      double g = config_.G;

      double b = k / l;
      double stripe_len = k + l + g;

      double profit_by_each_ec_set =
          (len_stripe_ - 1) / (blocks_per_node * (k / stripe_len) * b +
                               blocks_per_node * (l / stripe_len) * b +
                               blocks_per_node * (g / stripe_len) * k);

      profit_by_each_ec_set = profit_by_each_ec_set * 1024 * 1024 /
                              static_cast<double>(config_.block_size);

      std::cout << std::format("profit_by_each_ec_set：{}\n",
                               profit_by_each_ec_set);

      double expected_scatter_width = config_.expected_scatter_width;

      num_nodes_permutations_ =
          std::ceil(expected_scatter_width / profit_by_each_ec_set);
    } else if (TYPE == solution_type::copyset) {
      num_nodes_permutations_ =
          config_.expected_scatter_width / (len_stripe_ - 1);
      num_nodes_permutations_ +=
          (config_.expected_scatter_width % (len_stripe_ - 1) == 0 ? 0 : 1);
    } else {
      assert(false);
    }
  }

  double get_data_loss_probability_when_g_plus_2_fail() {
    double total_case = get_combination(config_.num_nodes, config_.G + 2);
    double failed_case = get_combination(len_stripe_, config_.G + 2) *
                         static_cast<double>(ec_sets_.size());
    return failed_case / total_case;
  }

  std::vector<int> get_repair_group(int block_id,
                                    const std::vector<int> &ec_set) {
    std::vector<int> repair_group{};
    if (block_id < config_.K) {
      int local_group_id = block_id / num_data_blocks_each_local_group_;
      int local_parity_block_id = config_.K + local_group_id;
      repair_group.push_back(ec_set[local_parity_block_id]);

      int index_block_id = num_data_blocks_each_local_group_ * local_group_id;
      for (int i = 0; i < num_data_blocks_each_local_group_; i++) {
        if (index_block_id != block_id) {
          repair_group.push_back(ec_set[index_block_id]);
        }
        index_block_id++;
      }
    } else if (block_id >= config_.K && block_id < config_.K + config_.L) {
      int local_group_id = block_id - config_.K;

      int index_block_id = num_data_blocks_each_local_group_ * local_group_id;
      for (int i = 0; i < num_data_blocks_each_local_group_; i++) {
        if (index_block_id != block_id) {
          repair_group.push_back(ec_set[index_block_id]);
        }
        index_block_id++;
      }
    } else if (block_id >= config_.K + config_.L) {
      for (int i = 0; i < config_.K; i++) {
        repair_group.push_back(ec_set[i]);
      }
    }
    return repair_group;
  }

  // 在一个条带中，先是K个数据块，接着是L个局部校验块，最后是G个全局校验块
  int len_stripe_;
  int num_data_blocks_each_local_group_;
  int num_nodes_permutations_;
  ec_set_solution_config config_;

  std::vector<int> nodes_permutation_;
  std::vector<std::vector<int>> repair_load_graph_;
  std::vector<std::vector<int>> ec_sets_{};
  std::map<int, std::unordered_set<int>> helpers_of_each_node_{};
  std::map<int, int> repair_IO_when_node_fail_{};
};

class copy_set_solution {
public:
private:
};

int main() {
  for (int expected_scatter_width = 10; expected_scatter_width <= 1000;
       expected_scatter_width += 10) {
    solution<solution_type::ecset> solution1(
        {.expected_scatter_width = expected_scatter_width,
         .K = 8,
         .L = 4,
         .G = 3,
         .num_nodes = 5000,
         .num_stripes = 1000000,
         .block_size = 128});
    solution<solution_type::copyset> solution2(
        {.expected_scatter_width = expected_scatter_width,
         .K = 8,
         .L = 4,
         .G = 3,
         .num_nodes = 5000,
         .num_stripes = 1000000,
         .block_size = 128});

    solution1.get_config().print_config();
    solution1.test_scatter_width();
    solution2.test_scatter_width();

    std::cout << std::endl;
  }

  return 0;
}