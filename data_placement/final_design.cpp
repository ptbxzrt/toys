#include <algorithm>
#include <cassert>
#include <cstddef>
#include <exception>
#include <format>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../tools.hpp"
#include "zipf.h"

enum class partition_strategy { RANDOM, ECWIDE, ICPP23 };

enum class selection_strategy { RANDOM, LOAD_BALANCE };

enum class placement_strategy {
  RANDOM_RANDOM = (static_cast<int>(partition_strategy::RANDOM) << 1) +
                  static_cast<int>(selection_strategy::RANDOM),
  ECWIDE_RANDOM = (static_cast<int>(partition_strategy::ECWIDE) << 1) +
                  static_cast<int>(selection_strategy::RANDOM),
  ICPP23_RANDOM = (static_cast<int>(partition_strategy::ICPP23) << 1) +
                  static_cast<int>(selection_strategy::RANDOM),
  ICPP23_LOAD_BALANCE = (static_cast<int>(partition_strategy::ICPP23) << 1) +
                        static_cast<int>(selection_strategy::LOAD_BALANCE)
};

partition_strategy get_partition_strategy(placement_strategy placement) {
  int value = (static_cast<int>(placement) >> 1);
  return static_cast<partition_strategy>(value);
}

selection_strategy get_selection_strategy(placement_strategy placement) {
  int value = static_cast<int>(placement) & 0x01;
  return static_cast<selection_strategy>(value);
}

void assert_enum() {
  assert(get_partition_strategy(placement_strategy::RANDOM_RANDOM) ==
         partition_strategy::RANDOM);
  assert(get_selection_strategy(placement_strategy::RANDOM_RANDOM) ==
         selection_strategy::RANDOM);

  assert(get_partition_strategy(placement_strategy::ECWIDE_RANDOM) ==
         partition_strategy::ECWIDE);
  assert(get_selection_strategy(placement_strategy::ECWIDE_RANDOM) ==
         selection_strategy::RANDOM);

  assert(get_partition_strategy(placement_strategy::ICPP23_RANDOM) ==
         partition_strategy::ICPP23);
  assert(get_selection_strategy(placement_strategy::ICPP23_RANDOM) ==
         selection_strategy::RANDOM);

  assert(get_partition_strategy(placement_strategy::ICPP23_LOAD_BALANCE) ==
         partition_strategy::ICPP23);
  assert(get_selection_strategy(placement_strategy::ICPP23_LOAD_BALANCE) ==
         selection_strategy::LOAD_BALANCE);
}

struct placement_config {
  int K = 8, L = 4, G = 3;

  int num_nodes = 5000;
  int num_clusters = 100;

  // 优秀cluster占比
  double better_cluster_factor = 0.5;
  int normal_cluster_bandwidth = 1;
  int better_cluster_bandwidth = 10;

  // 每个cluster中，优秀node占比
  double better_node_factor = 0.5;
  int normal_node_storage = 8;
  int better_node_storage = 32;

  double alpha = 0.5;
  double xigema = 0.2;
};

struct node_info {
  int node_id;
  int cluster_id;

  double storage;
  double storage_cost = 0;

  int num_data_blocks = 0;
  int num_local_parity_blocks = 0;
  int num_global_parity_blocks = 0;

  std::unordered_set<int> stripe_ids;
};

struct cluster_info {
  int cluster_id;

  double cluster_bandwidth;
  double storage;

  double network_cost = 0;
  double storage_cost = 0;

  std::vector<int> node_ids;
};

struct stripe_info {
  int stripe_id;
  int K, L, G;
  // 依次对应数据块、全局校验块、局部校验块
  std::vector<int> node_ids;
};

class data_placement {
public:
  data_placement(placement_config conf)
      : conf_(conf), shuffle_clusters_(std::mt19937(std::random_device{}())),
        shuffle_nodes_(std::mt19937(std::random_device{}())) {
    init_clusters_and_nodes();
    repair_load_distributions_node_cha_cluster.resize(
        nodes_info_.size(), std::vector<int>(clusters_info_.size(), 0));
    repair_load_distributions_cluster_cha_cluster.resize(
        clusters_info_.size(), std::vector<double>(clusters_info_.size(), 0));
    for (const auto &node : nodes_info_) {
      failed_node_ids_.push_back(node.node_id);
    }
    std::shuffle(failed_node_ids_.begin(), failed_node_ids_.end(),
                 shuffle_nodes_);
    failed_node_ids_.resize(nodes_info_.size() / 100);
  }

  static void test_case_load_balance() {
    // data_placement icpp23_random(placement_config{});
    data_placement icpp23_load_balance(placement_config{});

    int num_iterations = 100;
    for (int turn = 0; turn < num_iterations; turn++) {
      int num_stripes = 50000 / num_iterations;
      for (int i = 0; i < num_stripes; i++) {
        // icpp23_random.insert_one_stripe(placement_strategy::ICPP23_RANDOM);
        icpp23_load_balance.insert_one_stripe(
            placement_strategy::ICPP23_LOAD_BALANCE);
      }

      int num_reads = 950000 / num_iterations;
      std::default_random_engine generator;
      zipfian_int_distribution<int> zipf_dis(0, num_stripes - 1, 0.9);
      for (int i = 0; i < num_reads; i++) {
        int stripe_id = num_stripes - zipf_dis(generator) - 1;
        // icpp23_random.read_one_stripe_data(stripe_id);
        icpp23_load_balance.read_one_stripe_data(stripe_id);
      }
    }

    // icpp23_random.show_clusters_and_nodes();
    icpp23_load_balance.show_clusters_and_nodes();
  }

  static void test_case_loss_probability() {
    std::vector<int> num_clusters = {10, 20, 40, 50, 100, 125, 150};
    for (const auto &num_c : num_clusters) {
      std::cout << std::format("\nzhaoritian\n");
      double test_times = 100;
      int blocks_per_node = 50000;

      int count = 0;
      std::mutex m;

      std::vector<std::future<int>> results;
      for (int thr = 0; thr < 4; thr++) {
        results.push_back(std::async(std::launch::async, [&]() -> int {
          int failed_times = 0;
          while (true) {
            m.lock();
            if (count == test_times) {
              m.unlock();
              return failed_times;
            }
            count++;
            m.unlock();

            data_placement icpp23_random(
                placement_config{.num_clusters = num_c});
            std::unordered_set<int> failed_node_ids;
            for (const auto &node_id : icpp23_random.failed_node_ids_) {
              failed_node_ids.insert(node_id);
            }

            double num_all_blocks =
                static_cast<double>(icpp23_random.conf_.num_nodes) *
                static_cast<double>(blocks_per_node);
            double num_blocks_each_stripe = icpp23_random.conf_.K +
                                            icpp23_random.conf_.G +
                                            icpp23_random.conf_.L;
            std::uint64_t num_stripes = num_all_blocks / num_blocks_each_stripe;

            for (std::uint64_t i = 0; i < num_stripes; i++) {
              icpp23_random.insert_one_stripe(
                  placement_strategy::ICPP23_RANDOM);
              int stripe_id = icpp23_random.next_stripe_id_ - 1;
              std::vector<int> failed_node_ids_in_stripe;
              for (const auto &node_id :
                   icpp23_random.stripes_info_[stripe_id].node_ids) {
                if (failed_node_ids.contains(node_id)) {
                  failed_node_ids_in_stripe.push_back(node_id);
                }
              }
              if (failed_node_ids_in_stripe.size() >
                  icpp23_random.conf_.G + 1) {
                std::cout << std::format("\n[\n");
                std::cout << std::format(
                    "num_stripes：{}，num_nodes：{}，num_clusters：{}\n",
                    num_stripes, icpp23_random.conf_.num_nodes,
                    icpp23_random.conf_.num_clusters);
                std::cout << std::format("丢数据stripe {} 包含的节点：",
                                         stripe_id);
                print1DVector(icpp23_random.stripes_info_[stripe_id].node_ids);
                std::cout << std::format("条带中失败的节点: ");
                print1DVector(failed_node_ids_in_stripe);
                std::cout << std::format("整个集群失败的节点: ");
                print1DVector(icpp23_random.failed_node_ids_);
                std::cout << "]\n";
                failed_times++;
                break;
              }
            }
            // icpp23_random.show_clusters_and_nodes();
          }
          return failed_times;
        }));
      }

      double total_failed_times = 0;
      for (auto &f : results) {
        total_failed_times += static_cast<double>(f.get());
      }

      std::cout << std::format(
          "num_clusters：{}， 一共测试{}次，数据丢失{}次，比例为{}\n", num_c,
          test_times, total_failed_times, total_failed_times / test_times);
    }
  }

private:
  void insert_one_stripe(placement_strategy pla) {
    partition_strategy par = get_partition_strategy(pla);
    selection_strategy sel = get_selection_strategy(pla);

    std::vector<
        std::pair<std::vector<int>, std::unordered_map<int, std::vector<int>>>>
        partition_plan;
    switch (par) {
    case partition_strategy::RANDOM:
      partition_plan = partition_RANDOM();
      break;
    case partition_strategy::ECWIDE:
      partition_plan = partition_ECWIDE();
      break;
    case partition_strategy::ICPP23:
      partition_plan = partition_ICPP23();
      break;
    default:
      std::terminate();
    }

    int stripe_id = next_stripe_id_++;
    stripes_info_[stripe_id] = {
        .stripe_id = stripe_id, .K = conf_.K, .L = conf_.L, .G = conf_.G};
    stripe_info &stripe = stripes_info_[stripe_id];
    switch (sel) {
    case selection_strategy::RANDOM:
      selection_random(partition_plan, stripe_id, stripe.node_ids);
      break;
    case selection_strategy::LOAD_BALANCE:
      selection_rw_and_repair_load_balance(partition_plan, stripe_id,
                                           stripe.node_ids);
      break;
    default:
      std::terminate();
    }

    for (auto node_id : stripe.node_ids) {
      nodes_info_[node_id].storage_cost += 1;
      clusters_info_[nodes_info_[node_id].cluster_id].storage_cost += 1;
      clusters_info_[nodes_info_[node_id].cluster_id].network_cost += 1;
    }

    for (int i = 0; i < stripe.K + stripe.G + stripe.L; i++) {
      if (i < stripe.K) {
        nodes_info_[stripe.node_ids[i]].num_data_blocks++;
      } else if (i < stripe.K + stripe.G) {
        nodes_info_[stripe.node_ids[i]].num_global_parity_blocks++;
      } else {
        nodes_info_[stripe.node_ids[i]].num_local_parity_blocks++;
      }
    }

    // for (int failed_block_index = 0;
    //      failed_block_index < conf_.K + conf_.L + conf_.G;
    //      failed_block_index++) {
    //   node_info node = nodes_info_[stripe.node_ids[failed_block_index]];

    //   // 记录了本次修复涉及的cluster id
    //   std::vector<int> repair_span_cluster;
    //   //
    //   记录了需要从哪些cluster中,读取哪些block,记录顺序和repair_span_cluster对应
    //   // cluster_id->vector(node_id, block_index)
    //   std::vector<std::vector<std::pair<int, int>>>
    //       blocks_to_read_in_each_cluster;
    //   std::vector<std::pair<int, int>> new_locations_with_block_index;
    //   generate_repair_plan(stripe_id, failed_block_index,
    //                        blocks_to_read_in_each_cluster,
    //                        repair_span_cluster,
    //                        new_locations_with_block_index);
    //   for (const auto &cluster_id : repair_span_cluster) {
    //     if (cluster_id != node.cluster_id) {
    //       repair_load_distributions_node_cha_cluster[node.node_id]
    //                                                 [cluster_id] += 1;

    //       repair_load_distributions_cluster_cha_cluster[node.cluster_id]
    //                                                    [cluster_id] +=
    //           (1 / static_cast<double>(
    //                    clusters_info_[cluster_id].cluster_bandwidth));
    //     }
    //   }
    // }

    // std::cout << std::format("{} stripe inserted\n", stripe_id);
  }

  void read_one_stripe_data(int stripe_id) {
    stripe_info &stripe = stripes_info_[stripe_id];
    for (const auto &node_id : stripe.node_ids) {
      cluster_info &cluster = clusters_info_[nodes_info_[node_id].cluster_id];
      cluster.network_cost++;
    }
  }

  void init_clusters_and_nodes() {
    assert(conf_.num_nodes % conf_.num_clusters == 0);
    int num_nodes_per_cluster = conf_.num_nodes / conf_.num_clusters;

    int num_normal_clusters =
        conf_.num_clusters * (1 - conf_.better_cluster_factor);
    int num_better_clusters = conf_.num_clusters - num_normal_clusters;

    int num_normal_nodes_per_cluster =
        num_nodes_per_cluster * (1 - conf_.better_node_factor);
    int num_better_nodes_per_cluster =
        num_nodes_per_cluster - num_normal_nodes_per_cluster;

    int cluster_id = 0;
    int node_id = 0;
    for (int i = 0; i < conf_.num_clusters; i++) {
      int cluster_bandwidth = (i < num_normal_clusters)
                                  ? conf_.normal_cluster_bandwidth
                                  : conf_.better_cluster_bandwidth;
      clusters_info_.push_back(
          {.cluster_id = cluster_id,
           .cluster_bandwidth = static_cast<double>(cluster_bandwidth)});
      for (int j = 0; j < num_nodes_per_cluster; j++) {
        int storage;
        storage = (j < num_normal_nodes_per_cluster)
                      ? conf_.normal_node_storage
                      : conf_.better_node_storage;
        nodes_info_.push_back({.node_id = node_id,
                               .cluster_id = cluster_id,
                               .storage = static_cast<double>(storage)});
        clusters_info_[cluster_id].node_ids.push_back(node_id);
        clusters_info_[cluster_id].storage += static_cast<double>(storage);
        node_id++;
      }
      cluster_id++;
    }
  }

  void show_clusters_and_nodes() {
    for (const auto &cluster : clusters_info_) {
      std::cout << std::format("{}号集群，交换机带宽为{}，集群容量为{}。\n",
                               cluster.cluster_id, cluster.cluster_bandwidth,
                               cluster.storage);
      for (const auto &node_id : cluster.node_ids) {
        const node_info &node = nodes_info_[node_id];
        std::cout << std::format(
            "{:4}号节点，节点容量为{:4}。一共存储了{:8}个块，节点存储了{:8}"
            "个数据块，{:8}"
            "个全局校验块，{:8}"
            "个局部校验块。\n",
            node.node_id, node.storage, node.storage_cost, node.num_data_blocks,
            node.num_global_parity_blocks, node.num_local_parity_blocks);
      }
    }

    show_load_balance();

    show_repair_load_distribution_cross_cluster();
  }

  void show_load_balance() {
    double node_storage_load_balance = 0;
    {
      double max_storage_cost = std::numeric_limits<double>::min(),
             avg_storage_cost = 0;
      for (const auto &node : nodes_info_) {
        double storage_cost = node.storage_cost / node.storage;
        max_storage_cost = std::max(max_storage_cost, storage_cost);
        avg_storage_cost += storage_cost;
      }
      avg_storage_cost /= static_cast<double>(nodes_info_.size());
      node_storage_load_balance =
          (max_storage_cost - avg_storage_cost) / avg_storage_cost;
    }

    double cluster_network_load_balance = 0;
    {
      double max_network_cost = std::numeric_limits<double>::min(),
             avg_network_cost = 0;
      for (const auto &cluster : clusters_info_) {
        double network_cost = cluster.network_cost / cluster.cluster_bandwidth;
        max_network_cost = std::max(max_network_cost, network_cost);
        avg_network_cost += network_cost;
      }
      avg_network_cost /= static_cast<double>(clusters_info_.size());
      cluster_network_load_balance =
          (max_network_cost - avg_network_cost) / avg_network_cost;
    }
    std::cout << std::format(
        "集群间网络负载均衡情况{}，节点间存储负载均衡情况{}\n",
        cluster_network_load_balance, node_storage_load_balance);
  }

  void show_repair_load_distribution_cross_cluster() {
    for (std::size_t node_id = 0;
         node_id < repair_load_distributions_node_cha_cluster.size();
         node_id++) {
      const auto &load_in_each_cluster =
          repair_load_distributions_node_cha_cluster[node_id];
      std::cout << std::format("{:4}号节点宕机，一共产生{:6}跨集群IO：\n",
                               node_id,
                               std::accumulate(load_in_each_cluster.begin(),
                                               load_in_each_cluster.end(), 0));
      for (std::size_t cluster_id = 0; cluster_id < load_in_each_cluster.size();
           cluster_id++) {
        if (cluster_id != nodes_info_[node_id].cluster_id &&
            load_in_each_cluster[cluster_id] != 0) {
          std::cout << std::format(
              "----在{:4}号集群产生{:6}跨集群IO\n", cluster_id,
              static_cast<double>(load_in_each_cluster[cluster_id]) /
                  clusters_info_[cluster_id].cluster_bandwidth);
        }
      }
    }
  }

  std::vector<int> get_repair_load_distribution_cross_cluster(int node_id) {
    node_info node = nodes_info_[node_id];

    std::vector<int> distribution(clusters_info_.size(), 0);
    for (const auto &stripe_id : node.stripe_ids) {
      // 找到条带内的哪一个block损坏了
      int failed_block_index = -1;
      for (std::size_t i = 0; i < stripes_info_[stripe_id].node_ids.size();
           i++) {
        if (stripes_info_[stripe_id].node_ids[i] == node.node_id) {
          failed_block_index = i;
        }
      }
      assert(failed_block_index != -1);

      // 记录了本次修复涉及的cluster id
      std::vector<int> repair_span_cluster;
      // 记录了需要从哪些cluster中,读取哪些block,记录顺序和repair_span_cluster对应
      // cluster_id->vector(node_id, block_index)
      std::vector<std::vector<std::pair<int, int>>>
          blocks_to_read_in_each_cluster;
      std::vector<std::pair<int, int>> new_locations_with_block_index;
      generate_repair_plan(stripe_id, failed_block_index,
                           blocks_to_read_in_each_cluster, repair_span_cluster,
                           new_locations_with_block_index);
      for (const auto &cluster_id : repair_span_cluster) {
        if (cluster_id != node.cluster_id) {
          distribution[cluster_id] += 1;
        }
      }
    }

    return distribution;
  }

  void selection_random(
      std::vector<std::pair<std::vector<int>,
                            std::unordered_map<int, std::vector<int>>>>
          partition_plan,
      int stripe_id, std::vector<int> &selected_nodes) {

    selected_nodes.resize(stripes_info_[stripe_id].K +
                          stripes_info_[stripe_id].L +
                          stripes_info_[stripe_id].G);

    std::vector<int> cluster_ids;
    for (const auto &cluster : clusters_info_) {
      cluster_ids.push_back(cluster.cluster_id);
    }
    std::shuffle(cluster_ids.begin(), cluster_ids.end(), shuffle_clusters_);
    int cluster_idx = 0;

    for (std::size_t i = 0; i < partition_plan.size(); i++) {
      cluster_info &cluster = clusters_info_[cluster_ids[cluster_idx++]];

      std::vector<int> node_ids = cluster.node_ids;
      std::shuffle(node_ids.begin(), node_ids.end(), shuffle_nodes_);
      int node_idx = 0;

      auto find_a_node_for_a_block = [&, this](int block_idx) {
        selected_nodes[block_idx] = node_ids[node_idx];
        nodes_info_[node_ids[node_idx]].stripe_ids.insert(stripe_id);
        node_idx++;
      };

      for (int j = 0; j < partition_plan[i].first[0]; j++) {
        find_a_node_for_a_block(partition_plan[i].second[0][j]);
      }

      for (int j = 0; j < partition_plan[i].first[1]; j++) {
        find_a_node_for_a_block(partition_plan[i].second[1][j]);
      }

      for (int j = 0; j < partition_plan[i].first[2]; j++) {
        find_a_node_for_a_block(partition_plan[i].second[2][j]);
      }
    }
  }

  void selection_load_balance(
      std::vector<std::pair<std::vector<int>,
                            std::unordered_map<int, std::vector<int>>>>
          partition_plan,
      int stripe_id, std::vector<int> &selected_nodes) {

    selected_nodes.resize(stripes_info_[stripe_id].K +
                          stripes_info_[stripe_id].L +
                          stripes_info_[stripe_id].G);

    std::vector<int> cluster_ids;
    for (const auto &cluster : clusters_info_) {
      cluster_ids.push_back(cluster.cluster_id);
    }
    std::shuffle(cluster_ids.begin(), cluster_ids.end(), shuffle_clusters_);

    std::vector<double> num_of_blocks_each_par;
    std::vector<double> num_of_data_blocks_each_par;
    for (const auto &partition : partition_plan) {
      num_of_blocks_each_par.push_back(partition.first[0] + partition.first[1] +
                                       partition.first[2]);
      num_of_data_blocks_each_par.push_back(partition.first[0]);
    }

    double avg_blocks = 0;
    double avg_data_blocks = 0;
    for (std::size_t i = 0; i < partition_plan.size(); i++) {
      avg_blocks += num_of_blocks_each_par[i];
      avg_data_blocks += num_of_data_blocks_each_par[i];
    }
    avg_blocks =
        avg_blocks / static_cast<double>(num_of_blocks_each_par.size());
    avg_data_blocks = avg_data_blocks /
                      static_cast<double>(num_of_data_blocks_each_par.size());

    std::vector<std::pair<
        std::pair<std::vector<int>, std::unordered_map<int, std::vector<int>>>,
        double>>
        prediction_cost_each_par;
    for (std::size_t i = 0; i < partition_plan.size(); i++) {
      double storage_cost = num_of_blocks_each_par[i] / avg_blocks;
      double network_cost = num_of_data_blocks_each_par[i] / avg_data_blocks;
      double prediction_cost =
          storage_cost * (1 - conf_.alpha) + network_cost * conf_.alpha;
      prediction_cost_each_par.push_back({partition_plan[i], prediction_cost});
    }
    // 将partition按预计开销降序排列
    std::sort(prediction_cost_each_par.begin(), prediction_cost_each_par.end(),
              [](std::pair<std::pair<std::vector<int>,
                                     std::unordered_map<int, std::vector<int>>>,
                           double> &a,
                 std::pair<std::pair<std::vector<int>,
                                     std::unordered_map<int, std::vector<int>>>,
                           double> &b) { return a.second > b.second; });
    partition_plan.clear();
    for (const auto &partition : prediction_cost_each_par) {
      partition_plan.push_back(partition.first);
    }

    double cluster_avg_storage_cost, cluster_avg_network_cost;
    compute_avg_cost_for_each_cluster(cluster_avg_storage_cost,
                                      cluster_avg_network_cost);

    std::vector<std::pair<int, double>> sorted_clusters;
    for (const auto &cluster_id : cluster_ids) {
      cluster_info cluster = clusters_info_[cluster_id];
      double cluster_storage_cost = cluster.storage_cost / cluster.storage;
      double cluster_network_cost =
          cluster.network_cost / cluster.cluster_bandwidth;

      double combined_cost =
          (cluster_storage_cost / cluster_avg_storage_cost) *
              (1 - conf_.alpha) +
          (cluster_network_cost / cluster_avg_network_cost) * conf_.alpha;
      // 集群级别，根据复合负载排序
      sorted_clusters.push_back({cluster.cluster_id, combined_cost});
    }
    std::sort(sorted_clusters.begin(), sorted_clusters.end(),
              [](std::pair<int, double> &a, std::pair<int, double> &b) {
                return a.second < b.second;
              });

    int cluster_idx = 0;
    for (auto i = 0; i < partition_plan.size(); i++) {
      cluster_info &cluster =
          clusters_info_[sorted_clusters[cluster_idx++].first];

      std::vector<std::pair<int, double>> sorted_nodes_in_each_cluster;
      for (const auto &node_id : cluster.node_ids) {
        node_info &node = nodes_info_[node_id];
        double node_storage_cost = node.storage_cost / node.storage;
        // 节点级别，根据存储负载排序
        sorted_nodes_in_each_cluster.push_back({node_id, node_storage_cost});
      }
      std::sort(sorted_nodes_in_each_cluster.begin(),
                sorted_nodes_in_each_cluster.end(),
                [](std::pair<int, double> &a, std::pair<int, double> &b) {
                  return a.second < b.second;
                });

      std::vector<int> node_ids;
      for (const auto &pr : sorted_nodes_in_each_cluster) {
        node_ids.push_back(pr.first);
        if (node_ids.size() == partition_plan[i].first[0] +
                                   partition_plan[i].first[1] +
                                   partition_plan[i].first[2]) {
          break;
        }
      }
      std::shuffle(node_ids.begin(), node_ids.end(), shuffle_nodes_);

      int node_idx = 0;
      // data
      for (int j = 0; j < partition_plan[i].first[0]; j++) {
        int node_id = node_ids[node_idx++];
        selected_nodes[partition_plan[i].second[0][j]] = node_id;
        nodes_info_[node_id].stripe_ids.insert(stripe_id);
      }
      // local
      for (int j = 0; j < partition_plan[i].first[1]; j++) {
        int node_id = node_ids[node_idx++];
        selected_nodes[partition_plan[i].second[1][j]] = node_id;
        nodes_info_[node_id].stripe_ids.insert(stripe_id);
      }
      // global
      for (int j = 0; j < partition_plan[i].first[2]; j++) {
        int node_id = node_ids[node_idx++];
        selected_nodes[partition_plan[i].second[2][j]] = node_id;
        nodes_info_[node_id].stripe_ids.insert(stripe_id);
      }
    }
  }

  void selection_rw_and_repair_load_balance(
      std::vector<std::pair<std::vector<int>,
                            std::unordered_map<int, std::vector<int>>>>
          partition_plan,
      int stripe_id, std::vector<int> &selected_nodes) {
    selected_nodes.resize(stripes_info_[stripe_id].K +
                          stripes_info_[stripe_id].L +
                          stripes_info_[stripe_id].G);

    std::vector<int> cluster_ids;
    for (const auto &cluster : clusters_info_) {
      cluster_ids.push_back(cluster.cluster_id);
    }
    std::shuffle(cluster_ids.begin(), cluster_ids.end(), shuffle_clusters_);

    double avg_storage_cost_cluster = 0;
    for (const auto &cluster_id : cluster_ids) {
      cluster_info cluster = clusters_info_[cluster_id];
      avg_storage_cost_cluster += (cluster.storage_cost / cluster.storage);
    }
    avg_storage_cost_cluster /= static_cast<double>(cluster_ids.size());
    double max_storage_cost_cluster =
        (1 + conf_.xigema) * avg_storage_cost_cluster;

    std::vector<int> candidate_cluster_ids;
    std::vector<int> overload_cluster_ids;
    for (const auto &cluster_id : cluster_ids) {
      cluster_info cluster = clusters_info_[cluster_id];
      if ((cluster.storage_cost / cluster.storage) <=
          max_storage_cost_cluster) {
        candidate_cluster_ids.push_back(cluster.cluster_id);
      } else {
        overload_cluster_ids.push_back(cluster.cluster_id);
      }
    }
    for (const auto &cluster_id : overload_cluster_ids) {
      if (candidate_cluster_ids.size() >= partition_plan.size()) {
        break;
      }
      candidate_cluster_ids.push_back(cluster_id);
    }
    assert(candidate_cluster_ids.size() >= partition_plan.size());

    candidate_cluster_ids.clear();
    for (const auto &cluster : clusters_info_) {
      candidate_cluster_ids.push_back(cluster.cluster_id);
    }
    std::shuffle(candidate_cluster_ids.begin(), candidate_cluster_ids.end(),
                 shuffle_clusters_);
    if (stripe_id == 10000) {
      int a = 0;
      a++;
    }
    // double min_cost = std::numeric_limits<double>::max();
    // int initial_cluster = -1;
    // for (const auto &cluster_id : candidate_cluster_ids) {
    //   cluster_info cluster = clusters_info_[cluster_id];

    //   double cost = 0;
    //   for (const auto &other_id : candidate_cluster_ids) {
    //     cost += (repair_load_distributions_cluster_cha_cluster[cluster_id]
    //                                                           [other_id]);
    //   }

    //   if (min_cost > cost) {
    //     min_cost = cost;
    //     initial_cluster = cluster_id;
    //   }
    // }
    double min_cost = std::numeric_limits<double>::max();
    int initial_cluster_id1 = -1;
    int initial_cluster_id2 = -1;
    for (const auto &cluster_id1 : candidate_cluster_ids) {
      for (const auto &cluster_id2 : candidate_cluster_ids) {
        double cost = 0;
        if (cluster_id1 != cluster_id2) {
          cost = repair_load_distributions_cluster_cha_cluster[cluster_id1]
                                                              [cluster_id2] +
                 repair_load_distributions_cluster_cha_cluster[cluster_id2]
                                                              [cluster_id1];
          if (min_cost > cost) {
            min_cost = cost;
            initial_cluster_id1 = cluster_id1;
            initial_cluster_id2 = cluster_id2;
          }
        }
      }
    }
    std::unordered_set<int> P{initial_cluster_id1, initial_cluster_id2};
    while (P.size() < partition_plan.size()) {
      double min_cost = std::numeric_limits<double>::max();
      double new_cluster_id = -1;
      for (const auto &cluster_id : candidate_cluster_ids) {
        double cost = 0;
        if (P.contains(cluster_id) == false) {
          for (const auto &other_id : P) {
            cost += (repair_load_distributions_cluster_cha_cluster[cluster_id]
                                                                  [other_id] +
                     repair_load_distributions_cluster_cha_cluster[other_id]
                                                                  [cluster_id]);
          }
          if (min_cost > cost) {
            min_cost = cost;
            new_cluster_id = cluster_id;
          }
        }
      }
      P.insert(new_cluster_id);
    }
    candidate_cluster_ids.clear();
    candidate_cluster_ids.resize(P.size());
    std::copy(P.begin(), P.end(), candidate_cluster_ids.begin());
    std::shuffle(candidate_cluster_ids.begin(), candidate_cluster_ids.end(),
                 shuffle_clusters_);

    int cluster_idx = 0;
    for (std::size_t i = 0; i < partition_plan.size(); i++) {
      cluster_info &cluster =
          clusters_info_[candidate_cluster_ids[cluster_idx++]];
      std::vector<int> node_ids_in_cluster = cluster.node_ids;
      std::shuffle(node_ids_in_cluster.begin(), node_ids_in_cluster.end(),
                   shuffle_nodes_);

      double avg_storage_cost_node = 0;
      for (const auto &node_id : node_ids_in_cluster) {
        avg_storage_cost_node +=
            (nodes_info_[node_id].storage_cost / nodes_info_[node_id].storage);
      }
      avg_storage_cost_node /= static_cast<double>(node_ids_in_cluster.size());
      double max_storage_cost_node = (1 + conf_.xigema) * avg_storage_cost_node;

      std::vector<int> candidate_node_ids;
      std::vector<int> overload_node_ids;
      for (const auto &node_id : node_ids_in_cluster) {
        if ((nodes_info_[node_id].storage_cost /
             nodes_info_[node_id].storage) <= max_storage_cost_node) {
          candidate_node_ids.push_back(node_id);
        } else {
          overload_node_ids.push_back(node_id);
        }
      }
      int sum_blocks = partition_plan[i].first[0] + partition_plan[i].first[1] +
                       partition_plan[i].first[2];
      for (const auto &node_id : overload_node_ids) {
        if (candidate_node_ids.size() >= sum_blocks) {
          break;
        }
        candidate_node_ids.push_back(node_id);
      }
      assert(candidate_node_ids.size() >= sum_blocks);

      // std::vector<std::pair<int, int>> sum_distributions;
      // for (const auto &node_id : candidate_node_ids) {
      //   sum_distributions.push_back(
      //       {node_id,
      //        std::accumulate(repair_load_distributions_node_cha_cluster[node_id].begin(),
      //                        repair_load_distributions_node_cha_cluster[node_id].end(),
      //                        0)});
      // }
      // std::sort(sum_distributions.begin(), sum_distributions.end(),
      //           [](const std::pair<int, int> &a, const std::pair<int, int>
      //           &b) {
      //             return a.second < b.second;
      //           });

      std::vector<int> node_ids;
      for (const auto &pr : candidate_node_ids) {
        if (node_ids.size() == sum_blocks) {
          break;
        }
        node_ids.push_back(pr);
      }

      int node_idx = 0;
      // data
      for (int j = 0; j < partition_plan[i].first[0]; j++) {
        int node_id = node_ids[node_idx++];
        selected_nodes[partition_plan[i].second[0][j]] = node_id;
        nodes_info_[node_id].stripe_ids.insert(stripe_id);
      }
      // local
      for (int j = 0; j < partition_plan[i].first[1]; j++) {
        int node_id = node_ids[node_idx++];
        selected_nodes[partition_plan[i].second[1][j]] = node_id;
        nodes_info_[node_id].stripe_ids.insert(stripe_id);
      }
      // global
      for (int j = 0; j < partition_plan[i].first[2]; j++) {
        int node_id = node_ids[node_idx++];
        selected_nodes[partition_plan[i].second[2][j]] = node_id;
        nodes_info_[node_id].stripe_ids.insert(stripe_id);
      }
    }
  }

  // TODO
  std::vector<
      std::pair<std::vector<int>, std::unordered_map<int, std::vector<int>>>>
  partition_RANDOM() {
    std::vector<std::vector<int>> partition_plan(clusters_info_.size(),
                                                 std::vector<int>(3, 0));
    // (某个partition中已经包含的group编号, 某个partition中已经存放的块数量)
    std::vector<std::pair<std::unordered_set<int>, int>> help(
        clusters_info_.size());
    for (std::size_t i = 0; i < help.size(); i++) {
      help[i].second = 0;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, help.size() - 1);

    auto find_a_partition_for_a_block = [&, this]() {
      int partition_id;
      int space_upper;
      do {
        partition_id = dis(gen);
        space_upper = conf_.G + help[partition_id].first.size();
      } while (help[partition_id].second == space_upper);
      assert(help[partition_id].second < space_upper);

      help[partition_id].second++;
      return partition_id;
    };

    // 数据块
    int B = conf_.K / conf_.L;
    for (int i = 0; i < conf_.L; i++) {
      for (int j = 0; j < B; j++) {
        int partition_id = find_a_partition_for_a_block();
        help[partition_id].first.insert(i);
        partition_plan[partition_id][0]++;
      }
    }

    // 局部校验块
    for (int i = 0; i < conf_.L; i++) {
      int partition_id = find_a_partition_for_a_block();
      help[partition_id].first.insert(i);
      partition_plan[partition_id][1]++;
    }

    // 全局校验块
    for (int i = 0; i < conf_.G; i++) {
      int partition_id = find_a_partition_for_a_block();
      partition_plan[partition_id][2]++;
    }

    std::vector<std::vector<int>> partition_plan2;
    for (const auto &partition : partition_plan) {
      if (partition[0] != 0 || partition[1] != 0 || partition[2] != 0) {
        partition_plan2.push_back(partition);
      }
    }

    std::vector<
        std::pair<std::vector<int>, std::unordered_map<int, std::vector<int>>>>
        partition_plan3;
    for (const auto &partition : partition_plan2) {
      partition_plan3.push_back({partition, {}});
    }

    assert_partition(partition_plan3);

    return partition_plan3;
  }

  std::vector<
      std::pair<std::vector<int>, std::unordered_map<int, std::vector<int>>>>
  partition_ECWIDE() {
    auto partition_plan = partition_optimal_data_block();

    // ECWIDE将全局校验块单独放到了1个cluster
    partition_plan.push_back({{0, 0, conf_.G}, {}});
    for (int i = 0; i < conf_.G; i++) {
      partition_plan.back().second[2].push_back(conf_.K + i);
    }

    assert_partition(partition_plan);

    return partition_plan;
  }

  std::vector<
      std::pair<std::vector<int>, std::unordered_map<int, std::vector<int>>>>
  partition_ICPP23() {
    auto partition_plan = partition_optimal_data_block();

    std::vector<std::pair<int, int>> space_left_in_each_partition;
    int sum_left_space = 0;
    // 遍历每个partition查看剩余空间
    for (std::size_t i = 0; i < partition_plan.size(); i++) {
      int num_of_group = partition_plan[i].first[1];
      // 若某个partition不包含局部校验块,说明这里所有块属于1个group
      if (partition_plan[i].first[1] == 0) {
        num_of_group = 1;
      }
      int max_space = conf_.G + num_of_group;
      int left_space =
          max_space - partition_plan[i].first[0] - partition_plan[i].first[1];
      space_left_in_each_partition.push_back({i, left_space});
      sum_left_space += left_space;
    }

    // 用全局校验块填充剩余空间
    int left_g = conf_.G;
    if (sum_left_space >= conf_.G) {
      std::sort(space_left_in_each_partition.begin(),
                space_left_in_each_partition.end(),
                [](std::pair<int, int> &a, std::pair<int, int> &b) {
                  return a.second > b.second;
                });
      int global_parity_block_idx = conf_.K;
      for (std::size_t i = 0;
           i < space_left_in_each_partition.size() && left_g > 0; i++) {
        if (space_left_in_each_partition[i].second > 0) {
          int j = space_left_in_each_partition[i].first;
          if (left_g >= space_left_in_each_partition[i].second) {
            partition_plan[j].first[2] = space_left_in_each_partition[i].second;
            left_g -= partition_plan[j].first[2];
          } else {
            partition_plan[j].first[2] = left_g;
            left_g -= left_g;
          }
          for (int k = 0; k < partition_plan[j].first[2]; k++) {
            partition_plan[j].second[2].push_back(global_parity_block_idx);
            global_parity_block_idx++;
          }
        }
      }
      assert(left_g == 0);
    } else {
      partition_plan.push_back({{0, 0, left_g}, {}});

      int global_parity_block_idx = conf_.K;
      for (int i = 0; i < conf_.G; i++) {
        partition_plan.back().second[2].push_back(global_parity_block_idx);
        global_parity_block_idx++;
      }
    }

    assert_partition(partition_plan);

    return partition_plan;
  }

  std::vector<
      std::pair<std::vector<int>, std::unordered_map<int, std::vector<int>>>>
  partition_optimal_data_block() {
    // 表示分为3个partition
    // 第1个partition包含4个data block
    // 第2个partition包含2个local parity block
    // 第3个partition包含1个local parity block和1个global parity block
    // {
    //   {4, 0, 0},
    //   {0, 2, 0},
    //   {0, 1, 1}
    // }
    std::vector<
        std::pair<std::vector<int>, std::unordered_map<int, std::vector<int>>>>
        partition_plan;

    int B = conf_.K / conf_.L;
    int left_data_block_in_each_group = B;
    if (B >= conf_.G + 1) {
      for (int i = 0; i < conf_.L; i++) {
        int group_idx = i;
        int data_block_idx = group_idx * B;

        int num_of_left_data_block_in_cur_group = B;
        while (num_of_left_data_block_in_cur_group >= conf_.G + 1) {
          partition_plan.push_back({{conf_.G + 1, 0, 0}, {}});

          for (int j = 0; j < conf_.G + 1; j++) {
            partition_plan.back().second[0].push_back(data_block_idx);
            data_block_idx++;
          }

          num_of_left_data_block_in_cur_group -= (conf_.G + 1);
        }
        left_data_block_in_each_group = num_of_left_data_block_in_cur_group;
      }
    }

    assert(left_data_block_in_each_group == B % (conf_.G + 1));

    if (left_data_block_in_each_group == 0) {
      partition_plan.push_back({{0, conf_.L, 0}, {}});

      for (int i = 0; i < conf_.L; i++) {
        partition_plan.back().second[1].push_back(conf_.K + conf_.G + i);
      }
    } else {
      int theta = conf_.G / left_data_block_in_each_group;
      int num_of_left_group = conf_.L;
      int group_idx = 0;
      while (num_of_left_group >= theta) {
        partition_plan.push_back(
            {{theta * left_data_block_in_each_group, theta, 0}, {}});

        for (int i = 0; i < theta; i++) {
          int data_block_idx = group_idx * B + B - (B % (conf_.G + 1));
          for (int j = 0; j < left_data_block_in_each_group; j++) {
            partition_plan.back().second[0].push_back(data_block_idx);
            data_block_idx++;
          }
          partition_plan.back().second[1].push_back(conf_.K + conf_.G +
                                                    group_idx);
          group_idx++;
        }
        num_of_left_group -= theta;
      }
      if (num_of_left_group > 0) {
        partition_plan.push_back(
            {{num_of_left_group * left_data_block_in_each_group,
              num_of_left_group, 0},
             {}});

        for (int i = 0; i < num_of_left_group; i++) {
          int data_block_idx = group_idx * B + B - (B % (conf_.G + 1));
          for (int j = 0; j < left_data_block_in_each_group; j++) {
            partition_plan.back().second[0].push_back(data_block_idx);
            data_block_idx++;
          }
          partition_plan.back().second[1].push_back(conf_.K + conf_.G +
                                                    group_idx);
          group_idx++;
        }
      }
    }

    return partition_plan;
  }

  void assert_partition(
      const std::vector<std::pair<std::vector<int>,
                                  std::unordered_map<int, std::vector<int>>>>
          &partition_plan) {
    // 确保partition合法
    int sum = 0;
    for (const auto &partition : partition_plan) {
      for (auto num_block : partition.first) {
        sum += num_block;
      }
    }
    assert(sum == conf_.K + conf_.G + conf_.L);

    for (auto partition : partition_plan) {
      assert(partition.first[0] + partition.first[1] + partition.first[2] ==
             partition.second[0].size() + partition.second[1].size() +
                 partition.second[2].size());
    }

    std::vector<int> block_idxs;
    for (const auto &partition : partition_plan) {
      for (const auto &idxs : partition.second) {
        for (const auto &idx : idxs.second) {
          block_idxs.push_back(idx);
          if (idxs.first == 0) {
            assert(idx < conf_.K);
          } else if (idxs.first == 1) {
            assert(idx >= conf_.K + conf_.G);
          } else if (idxs.first == 2) {
            assert(idx >= conf_.K && idx < conf_.K + conf_.G);
          }
        }
      }
    }
    std::sort(block_idxs.begin(), block_idxs.end());
    for (int i = 0; i < conf_.K + conf_.G + conf_.L; i++) {
      assert(i == block_idxs[i]);
    }
  }

  void generate_repair_plan(
      int stripe_id, int failed_block_index,
      std::vector<std::vector<std::pair<int, int>>>
          &blocks_to_read_in_each_cluster,
      std::vector<int> &repair_span_cluster,
      std::vector<std::pair<int, int>> &new_locations_with_block_index) {
    stripe_info &stripe = stripes_info_[stripe_id];
    int k = stripe.K;
    int real_l = stripe.L;
    int g = stripe.G;
    int b = stripe.K / stripe.L;

    node_info &failed_node = nodes_info_[stripe.node_ids[failed_block_index]];
    int main_cluster_id = failed_node.cluster_id;
    repair_span_cluster.push_back(main_cluster_id);

    // 将修复好的块放回原位
    new_locations_with_block_index.push_back(
        {failed_node.node_id, failed_block_index});

    assert(failed_block_index >= 0 &&
           failed_block_index <= (k + g + real_l - 1));
    if (failed_block_index >= k && failed_block_index <= (k + g - 1)) {
      // 修复全局校验块
      std::unordered_map<int, std::vector<int>> live_blocks_in_each_cluster;
      // 找到每个cluster中的存活块
      for (int i = 0; i <= (k + g - 1); i++) {
        if (i != failed_block_index) {
          node_info &live_node = nodes_info_[stripe.node_ids[i]];
          live_blocks_in_each_cluster[live_node.cluster_id].push_back(i);
        }
      }

      std::unordered_map<int, std::vector<int>>
          live_blocks_needed_in_each_cluster;
      int num_of_needed_live_blocks = k;
      // 优先读取main cluster的,即损坏块所在cluster
      for (auto live_block_index :
           live_blocks_in_each_cluster[main_cluster_id]) {
        if (num_of_needed_live_blocks <= 0) {
          break;
        }
        live_blocks_needed_in_each_cluster[main_cluster_id].push_back(
            live_block_index);
        num_of_needed_live_blocks--;
      }

      // 需要对剩下的cluster中存活块的数量进行排序,优先从存活块数量多的cluster中读取
      std::vector<std::pair<int, std::vector<int>>>
          sorted_live_blocks_in_each_cluster;
      for (auto &cluster : live_blocks_in_each_cluster) {
        if (cluster.first != main_cluster_id) {
          sorted_live_blocks_in_each_cluster.push_back(
              {cluster.first, cluster.second});
        }
      }
      std::sort(sorted_live_blocks_in_each_cluster.begin(),
                sorted_live_blocks_in_each_cluster.end(),
                [](std::pair<int, std::vector<int>> &a,
                   std::pair<int, std::vector<int>> &b) {
                  return a.second.size() > b.second.size();
                });
      for (auto &cluster : sorted_live_blocks_in_each_cluster) {
        for (auto &block_index : cluster.second) {
          if (num_of_needed_live_blocks <= 0) {
            break;
          }
          live_blocks_needed_in_each_cluster[cluster.first].push_back(
              block_index);
          num_of_needed_live_blocks--;
        }
      }

      // 记录需要从main cluster中读取的存活块
      std::vector<std::pair<int, int>> blocks_to_read_in_main_cluster;
      for (auto &block_index :
           live_blocks_needed_in_each_cluster[main_cluster_id]) {
        node_info &node = nodes_info_[stripe.node_ids[block_index]];
        blocks_to_read_in_main_cluster.push_back({node.node_id, block_index});
      }
      blocks_to_read_in_each_cluster.push_back(blocks_to_read_in_main_cluster);

      // 记录需要从其它cluster中读取的存活块
      for (auto &cluster : live_blocks_needed_in_each_cluster) {
        if (cluster.first != main_cluster_id) {
          repair_span_cluster.push_back(cluster.first);

          std::vector<std::pair<int, int>> blocks_to_read_in_another_cluster;
          for (auto &block_index : cluster.second) {
            node_info &node = nodes_info_[stripe.node_ids[block_index]];
            blocks_to_read_in_another_cluster.push_back(
                {node.node_id, block_index});
          }
          blocks_to_read_in_each_cluster.push_back(
              blocks_to_read_in_another_cluster);
        }
      }
    } else {
      // 修复数据块和局部校验块
      int group_index = -1;
      if (failed_block_index >= 0 && failed_block_index <= (k - 1)) {
        group_index = failed_block_index / b;
      } else {
        group_index = failed_block_index - (k + g);
      }

      std::vector<std::pair<int, int>> live_blocks_in_group;
      for (int i = 0; i < b; i++) {
        int block_index = group_index * b + i;
        if (block_index != failed_block_index) {
          if (block_index >= k) {
            break;
          }
          live_blocks_in_group.push_back(
              {stripe.node_ids[block_index], block_index});
        }
      }
      if (failed_block_index != k + g + group_index) {
        live_blocks_in_group.push_back(
            {stripe.node_ids[k + g + group_index], k + g + group_index});
      }

      std::unordered_set<int> span_cluster;
      for (auto &live_block : live_blocks_in_group) {
        span_cluster.insert(nodes_info_[live_block.first].cluster_id);
      }
      for (auto &cluster_involved : span_cluster) {
        if (cluster_involved != main_cluster_id) {
          repair_span_cluster.push_back(cluster_involved);
        }
      }

      for (auto &cluster_id : repair_span_cluster) {
        std::vector<std::pair<int, int>> blocks_to_read_in_cur_cluster;
        for (auto &live_block : live_blocks_in_group) {
          node_info &node = nodes_info_[live_block.first];
          if (node.cluster_id == cluster_id) {
            blocks_to_read_in_cur_cluster.push_back(
                {node.node_id, live_block.second});
          }
        }
        if (cluster_id == main_cluster_id) {
          assert(blocks_to_read_in_cur_cluster.size() > 0);
        }
        blocks_to_read_in_each_cluster.push_back(blocks_to_read_in_cur_cluster);
      }
    }
  }

  void compute_avg_cost_for_each_cluster(double &cluster_avg_storage_cost,
                                         double &cluster_avg_network_cost) {
    for (auto &cluster : clusters_info_) {
      double storage_cost = cluster.storage_cost / cluster.storage;
      double network_cost = cluster.network_cost / cluster.cluster_bandwidth;

      cluster_avg_storage_cost += storage_cost;
      cluster_avg_network_cost += network_cost;
    }
    cluster_avg_storage_cost /= static_cast<double>(clusters_info_.size());
    cluster_avg_network_cost /= static_cast<double>(clusters_info_.size());
  }

  placement_config conf_;

  std::vector<cluster_info> clusters_info_;
  std::vector<node_info> nodes_info_;
  std::unordered_map<int, stripe_info> stripes_info_;
  std::vector<std::vector<int>> repair_load_distributions_node_cha_cluster;
  std::vector<std::vector<double>>
      repair_load_distributions_cluster_cha_cluster;
  std::vector<int> failed_node_ids_;

  int next_stripe_id_ = 0;

  std::mt19937 shuffle_clusters_;
  std::mt19937 shuffle_nodes_;
};

int main() {
  assert_enum();

  std::cout << std::format("cpu有{}个核\n",
                           std::thread::hardware_concurrency());

  // data_placement::test_case_load_balance();
  data_placement::test_case_loss_probability();

  return 0;
}