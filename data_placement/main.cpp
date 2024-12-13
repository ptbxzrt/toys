#include "../tools.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class partition_strategy { random, ECWIDE, ICPP23 };
enum class placement_strategy { random, ECWIDE, ICPP23_1, ICPP23_2 };

struct placement_config {
  int K, L, G;
  int num_stripes;

  int num_nodes;
  int num_clusters;

  // 优秀cluster数量
  double heterogeneous_factor1 = 0.5;
  // 每个cluster中优秀node数量
  double heterogeneous_factor2 = 0.5;

  double alpha = 0.5;
};

struct node_info {
  int node_id;
  int cluster_id;

  double storage;
  double bandwidth;

  double storage_cost;
  double network_cost;

  int num_data_blocks = 0;
  int num_local_parity_blocks = 0;
  int num_global_parity_blocks = 0;

  std::unordered_set<int> stripe_ids;

  void print_info() const {
    std::cout << std::format("Node ID: {:6}, Cluster ID: {:4}, Storage: {:4}, "
                             "Bandwidth: {:4}\n",
                             node_id, cluster_id, storage, bandwidth);
  }
};

struct cluster_info {
  int cluster_id;
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
  data_placement(const placement_config &conf)
      : conf_(conf), clusters_info_(conf_.num_clusters),
        nodes_info_(conf_.num_nodes) {
    std::vector<int> storage, bandwidth;
    init_storage_and_bandwidth(storage, bandwidth);

    assert(conf_.K + conf_.L + conf_.G <= conf_.num_nodes);
    assert(conf_.num_nodes % conf_.num_clusters == 0);
    assert(bandwidth.size() == storage.size() &&
           bandwidth.size() == conf_.num_nodes);

    int idx_cluster_id = 0;
    int idx_node_id = 0;
    int num_nodes_per_cluster = conf_.num_nodes / conf_.num_clusters;
    for (std::size_t i = 0; i < clusters_info_.size(); i++) {
      int cluster_id = idx_cluster_id++;
      clusters_info_[cluster_id].cluster_id = cluster_id;
      for (int j = 0; j < num_nodes_per_cluster; j++) {
        int node_id = idx_node_id++;
        nodes_info_[node_id].node_id = node_id;
        nodes_info_[node_id].cluster_id = cluster_id;
        nodes_info_[node_id].storage = storage[node_id];
        nodes_info_[node_id].bandwidth = bandwidth[node_id];

        clusters_info_[cluster_id].node_ids.push_back(node_id);
      }
    }
  }

  // 测试partition
  static void test_case_partition() {
    data_placement test_case({.K = 24,
                              .L = 4,
                              .G = 3,

                              .num_nodes = 1000,
                              .num_clusters = 10});
    test_case.show_partition(partition_strategy::random);
    test_case.show_partition(partition_strategy::ECWIDE);
    test_case.show_partition(partition_strategy::ICPP23);
  }

  // 测试placement
  static void test_case_placement() {
    data_placement test_case({.K = 8,
                              .L = 4,
                              .G = 3,

                              .num_nodes = 1000,
                              .num_clusters = 10});
    test_case.show_placement(placement_strategy::random);
    test_case.show_placement(placement_strategy::ECWIDE);
    test_case.show_placement(placement_strategy::ICPP23_1);
    test_case.show_placement(placement_strategy::ICPP23_2);
  }

  // 测试节点配置信息
  static void test_case_nodes_info() {
    data_placement test_case({.K = 8,
                              .L = 4,
                              .G = 3,

                              .num_nodes = 1000,
                              .num_clusters = 10});
    test_case.show_nodes_info();
  }

  // 测试load balance
  static void test_case_load_balance() {
    int num_stripes = 1000000;
    int K = 8, L = 4, G = 3;
    int num_nodes = 200;
    int num_clusters = 10;
    data_placement random({.K = K,
                           .L = L,
                           .G = G,
                           .num_stripes = num_stripes,

                           .num_nodes = num_nodes,
                           .num_clusters = num_clusters});
    data_placement ecwide({.K = K,
                           .L = L,
                           .G = G,
                           .num_stripes = num_stripes,

                           .num_nodes = num_nodes,
                           .num_clusters = num_clusters});
    data_placement icpp23_1({.K = K,
                             .L = L,
                             .G = G,
                             .num_stripes = num_stripes,

                             .num_nodes = num_nodes,
                             .num_clusters = num_clusters});
    data_placement icpp23_2({.K = K,
                             .L = L,
                             .G = G,
                             .num_stripes = num_stripes,

                             .num_nodes = num_nodes,
                             .num_clusters = num_clusters});

    random.insert_stripes(placement_strategy::random);
    std::cout << std::format("placement_strategy {}\n",
                             "placement_strategy::random");
    random.show_load_balance();

    ecwide.insert_stripes(placement_strategy::ECWIDE);
    std::cout << std::format("placement_strategy {}\n",
                             "placement_strategy::ECWIDE");
    ecwide.show_load_balance();

    icpp23_1.insert_stripes(placement_strategy::ICPP23_1);
    std::cout << std::format("placement_strategy {}\n",
                             "placement_strategy::ICPP23_1");
    icpp23_1.show_load_balance();

    icpp23_2.insert_stripes(placement_strategy::ICPP23_2);
    std::cout << std::format("placement_strategy {}\n",
                             "placement_strategy::ICPP23_2");
    icpp23_2.show_load_balance();
  }

  // 测试repair load distribution
  static void test_case_repair_load_distribution() {
    int num_stripes = 1000000;
    int num_nodes = 231;
    int num_clusters = 11;

    std::cout << std::format("placement_strategy {}\n",
                             "placement_strategy::ICPP23_2");
    data_placement ICPP23_2({.K = 8,
                             .L = 4,
                             .G = 3,
                             .num_stripes = num_stripes,

                             .num_nodes = num_nodes,
                             .num_clusters = num_clusters});
    ICPP23_2.insert_stripes(placement_strategy::ICPP23_2);
    for (auto node_info : ICPP23_2.nodes_info_) {
      std::cout << std::format("{:4}号节点，数据块数量{:6}，局部校验块数量{:"
                               "6}，全局校验块数量{}\n",
                               node_info.node_id, node_info.num_data_blocks,
                               node_info.num_local_parity_blocks,
                               node_info.num_global_parity_blocks);
    }

    for (int node_id = 0; node_id < num_nodes; node_id++) {
      ICPP23_2.show_repair_load_distribution(node_id);
    }
    // for (int node_id = num_nodes - 1; node_id >= 0; node_id--) {
    //   std::cout << std::format("placement_strategy {}\n",
    //                            "placement_strategy::ICPP23_1");
    //   data_placement ICPP23_1({.K = 16,
    //                            .L = 4,
    //                            .G = 4,
    //                            .num_stripes = num_stripes,

    //                            .num_nodes = 200,
    //                            .num_clusters = num_clusters});
    //   ICPP23_1.insert_stripes(placement_strategy::ICPP23_1);
    //   ICPP23_1.show_repair_load_distribution(node_id);
    // }
  }

  // 测试数据丢失概率
  static void test_case_data_loss_probability() {
    int num_stripes = 1000000;
    std::cout << std::format("placement_strategy {}\n",
                             "placement_strategy::random");
    show_data_loss_probability({.K = 8,
                                .L = 4,
                                .G = 3,
                                .num_stripes = num_stripes,

                                .num_nodes = 60,
                                .num_clusters = 10},
                               10, placement_strategy::ICPP23_1);
  }

private:
  // data placement = partition + selection
  std::vector<std::vector<int>> partition_random() {
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

    check_partition_plan(partition_plan2);
    return partition_plan2;
  }

  std::vector<std::vector<int>> partition_ECWIDE() {
    auto partition_plan = partition_optimal_data_block();

    // ECWIDE将全局校验块单独放到了1个cluster
    partition_plan.push_back({0, 0, conf_.G});

    check_partition_plan(partition_plan);
    return partition_plan;
  }

  std::vector<std::vector<int>> partition_ICPP23() {
    auto partition_plan = partition_optimal_data_block();

    std::vector<std::pair<int, int>> space_left_in_each_partition;
    int sum_left_space = 0;
    // 遍历每个partition查看剩余空间
    for (std::size_t i = 0; i < partition_plan.size(); i++) {
      int num_of_group = partition_plan[i][1];
      // 若某个partition不包含局部校验块,说明这里所有块属于1个group
      if (partition_plan[i][1] == 0) {
        num_of_group = 1;
      }
      int max_space = conf_.G + num_of_group;
      int left_space = max_space - partition_plan[i][0] - partition_plan[i][1];
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
      for (std::size_t i = 0;
           i < space_left_in_each_partition.size() && left_g > 0; i++) {
        if (space_left_in_each_partition[i].second > 0) {
          int j = space_left_in_each_partition[i].first;
          if (left_g >= space_left_in_each_partition[i].second) {
            partition_plan[j][2] = space_left_in_each_partition[i].second;
            left_g -= partition_plan[j][2];
          } else {
            partition_plan[j][2] = left_g;
            left_g -= left_g;
          }
        }
      }
      assert(left_g == 0);
    } else {
      partition_plan.push_back({0, 0, left_g});
    }

    check_partition_plan(partition_plan);
    return partition_plan;
  }

  // selected_nodes对应block顺序：数据块、全局校验块、局部校验块
  void selection_random(std::vector<std::vector<int>> partition_plan,
                        int stripe_id, std::vector<int> &selected_nodes) {
    selected_nodes.resize(stripes_info_[stripe_id].K +
                          stripes_info_[stripe_id].L +
                          stripes_info_[stripe_id].G);

    int data_block_idx = 0;
    int global_block_idx = stripes_info_[stripe_id].K;
    int local_block_idx =
        stripes_info_[stripe_id].K + stripes_info_[stripe_id].G;

    std::random_device rd_cluster;
    std::mt19937 gen_cluster(rd_cluster());
    std::uniform_int_distribution<int> dis_cluster(0,
                                                   clusters_info_.size() - 1);

    std::vector<bool> visited_clusters(clusters_info_.size(), false);

    for (std::size_t i = 0; i < partition_plan.size(); i++) {
      int cluster_id;
      do {
        cluster_id = dis_cluster(gen_cluster);
      } while (visited_clusters[cluster_id] == true);
      visited_clusters[cluster_id] = true;
      cluster_info &cluster = clusters_info_[cluster_id];

      std::random_device rd_node;
      std::mt19937 gen_node(rd_node());
      std::uniform_int_distribution<int> dis_node(0,
                                                  cluster.node_ids.size() - 1);

      std::vector<bool> visited_nodes(cluster.node_ids.size(), false);

      auto find_a_node_for_a_block = [&, this](int &block_idx) {
        int node_idx;
        do {
          // 注意,此处是node_idx,而非node_id
          // cluster.node_ids[node_idx]才是node_id
          node_idx = dis_node(gen_node);
        } while (visited_nodes[node_idx] == true);
        visited_nodes[node_idx] = true;
        selected_nodes[block_idx++] = cluster.node_ids[node_idx];
        nodes_info_[cluster.node_ids[node_idx]].stripe_ids.insert(stripe_id);
      };

      for (int j = 0; j < partition_plan[i][0]; j++) {
        find_a_node_for_a_block(data_block_idx);
      }

      for (int j = 0; j < partition_plan[i][1]; j++) {
        find_a_node_for_a_block(local_block_idx);
      }

      for (int j = 0; j < partition_plan[i][2]; j++) {
        find_a_node_for_a_block(global_block_idx);
      }
    }
  }

  void selection_load_balance(std::vector<std::vector<int>> partition_plan,
                              int stripe_id, std::vector<int> &selected_nodes) {
    selected_nodes.resize(stripes_info_[stripe_id].K +
                          stripes_info_[stripe_id].L +
                          stripes_info_[stripe_id].G);

    std::vector<double> num_of_blocks_each_par;
    std::vector<double> num_of_data_blocks_each_par;
    for (const auto &partition : partition_plan) {
      num_of_blocks_each_par.push_back(partition[0] + partition[1] +
                                       partition[2]);
      num_of_data_blocks_each_par.push_back(partition[0]);
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

    std::vector<std::pair<std::vector<int>, double>> prediction_cost_each_par;
    for (std::size_t i = 0; i < partition_plan.size(); i++) {
      double storage_cost = num_of_blocks_each_par[i] / avg_blocks;
      double network_cost = num_of_data_blocks_each_par[i] / avg_data_blocks;
      double prediction_cost =
          storage_cost * (1 - conf_.alpha) + network_cost * conf_.alpha;
      prediction_cost_each_par.push_back({partition_plan[i], prediction_cost});
    }
    // 将partition按预计开销降序排列
    std::sort(prediction_cost_each_par.begin(), prediction_cost_each_par.end(),
              [](std::pair<std::vector<int>, double> &a,
                 std::pair<std::vector<int>, double> &b) {
                return a.second > b.second;
              });
    partition_plan.clear();
    for (const auto &partition : prediction_cost_each_par) {
      partition_plan.push_back(partition.first);
    }

    int data_block_idx = 0;
    int global_block_idx = stripes_info_[stripe_id].K;
    int local_block_idx =
        stripes_info_[stripe_id].K + stripes_info_[stripe_id].G;

    double node_avg_storage_cost, node_avg_network_cost;
    double cluster_avg_storage_cost, cluster_avg_network_cost;
    compute_avg_cost_for_each_node_and_cluster(
        node_avg_storage_cost, node_avg_network_cost, cluster_avg_storage_cost,
        cluster_avg_network_cost);

    std::vector<std::pair<int, double>> sorted_clusters;
    for (auto &cluster : clusters_info_) {
      double cluster_storage_cost, cluster_network_cost;
      compute_total_cost_for_cluster(cluster, cluster_storage_cost,
                                     cluster_network_cost);
      double combined_cost =
          (cluster_storage_cost / cluster_avg_storage_cost) *
              (1 - conf_.alpha) +
          (cluster_network_cost / cluster_avg_network_cost) * conf_.alpha;
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
      for (auto &node_id : cluster.node_ids) {
        node_info &node = nodes_info_[node_id];
        double node_storage_cost = node.storage_cost / node.storage;
        double node_network_cost = node.network_cost / node.bandwidth;
        double combined_cost =
            (node_storage_cost / node_avg_storage_cost) * (1 - conf_.alpha) +
            (node_network_cost / node_avg_network_cost) * conf_.alpha;
        sorted_nodes_in_each_cluster.push_back({node_id, combined_cost});
      }
      std::sort(sorted_nodes_in_each_cluster.begin(),
                sorted_nodes_in_each_cluster.end(),
                [](std::pair<int, double> &a, std::pair<int, double> &b) {
                  return a.second < b.second;
                });

      int node_idx = 0;
      // data
      for (int j = 0; j < partition_plan[i][0]; j++) {
        int node_id = sorted_nodes_in_each_cluster[node_idx++].first;
        selected_nodes[data_block_idx++] = node_id;
        nodes_info_[node_id].stripe_ids.insert(stripe_id);
      }
      // local
      for (int j = 0; j < partition_plan[i][1]; j++) {
        int node_id = sorted_nodes_in_each_cluster[node_idx++].first;
        selected_nodes[local_block_idx++] = node_id;
        nodes_info_[node_id].stripe_ids.insert(stripe_id);
      }
      // global
      for (int j = 0; j < partition_plan[i][2]; j++) {
        int node_id = sorted_nodes_in_each_cluster[node_idx++].first;
        selected_nodes[global_block_idx++] = node_id;
        nodes_info_[node_id].stripe_ids.insert(stripe_id);
      }
    }
  }

  void placement_random(int stripe_id, std::vector<int> &selected_nodes) {
    selection_random(partition_random(), stripe_id, selected_nodes);
  }

  void placement_ECWIDE(int stripe_id, std::vector<int> &selected_nodes) {
    selection_random(partition_ECWIDE(), stripe_id, selected_nodes);
  }

  void placement_ICPP23_1(int stripe_id, std::vector<int> &selected_nodes) {
    selection_random(partition_ICPP23(), stripe_id, selected_nodes);
  }

  void placement_ICPP23_2(int stripe_id, std::vector<int> &selected_nodes) {
    selection_load_balance(partition_ICPP23(), stripe_id, selected_nodes);
  }

  // 下面的函数都是一些帮助函数
  std::vector<std::vector<int>> partition_optimal_data_block() {
    // 表示分为3个partition
    // 第1个partition包含4个data block
    // 第2个partition包含4个data block
    // 第3个partition包含1个local parity block和1个global parity block
    // {
    //   {4, 0, 0},
    //   {4, 0, 0},
    //   {0, 1, 1}
    // }
    std::vector<std::vector<int>> partition_plan;

    int B = conf_.K / conf_.L;
    int left_data_block_in_each_group = B;
    if (B >= conf_.G + 1) {
      for (int i = 0; i < conf_.L; i++) {
        int num_of_left_data_block_in_cur_group = B;
        while (num_of_left_data_block_in_cur_group >= conf_.G + 1) {
          partition_plan.push_back({conf_.G + 1, 0, 0});
          num_of_left_data_block_in_cur_group -= (conf_.G + 1);
        }
        left_data_block_in_each_group = num_of_left_data_block_in_cur_group;
      }
    }

    assert(left_data_block_in_each_group == B % (conf_.G + 1));

    if (left_data_block_in_each_group == 0) {
      partition_plan.push_back({0, conf_.L, 0});
    } else {
      int theta = conf_.G / left_data_block_in_each_group;
      int num_of_left_group = conf_.L;
      while (num_of_left_group >= theta) {
        partition_plan.push_back(
            {theta * left_data_block_in_each_group, theta, 0});
        num_of_left_group -= theta;
      }
      if (num_of_left_group > 0) {
        partition_plan.push_back(
            {num_of_left_group * left_data_block_in_each_group,
             num_of_left_group, 0});
      }
    }

    return partition_plan;
  }

  void
  check_partition_plan(const std::vector<std::vector<int>> &partition_plan) {
    int sum = 0;
    for (const auto &partition : partition_plan) {
      for (auto num_block : partition) {
        sum += num_block;
      }
    }
    assert(sum == conf_.K + conf_.G + conf_.L);
  }

  void init_storage_and_bandwidth(std::vector<int> &storage,
                                  std::vector<int> &bandwidth) {
    int num_nodes_per_cluster = conf_.num_nodes / conf_.num_clusters;

    int normal_storage = 8, better_storage = 32;
    int normal_bandwidth = 10, better_bandwidth = 40;

    int num_normal_clusters =
        conf_.num_clusters * (1 - conf_.heterogeneous_factor1);
    int num_better_clusters = conf_.num_clusters - num_normal_clusters;

    int num_normal_nodes_per_cluster =
        num_nodes_per_cluster * (1 - conf_.heterogeneous_factor2);
    int num_better_nodes_per_cluster =
        num_nodes_per_cluster - num_normal_nodes_per_cluster;

    for (int i = 0; i < num_normal_clusters; i++) {
      for (int j = 0; j < num_normal_nodes_per_cluster; j++) {
        storage.push_back(normal_storage);
        bandwidth.push_back(normal_bandwidth);
      }
      for (int j = 0; j < num_better_nodes_per_cluster; j++) {
        storage.push_back(better_storage);
        bandwidth.push_back(better_bandwidth);
      }
    }

    for (int i = 0; i < num_better_clusters; i++) {
      for (int j = 0; j < num_nodes_per_cluster; j++) {
        storage.push_back(better_storage);
        bandwidth.push_back(better_bandwidth);
      }
    }
  }

  void compute_total_cost_for_cluster(cluster_info &cluster,
                                      double &storage_cost,
                                      double &network_cost) {
    double all_storage = 0, all_bandwidth = 0;
    double all_storage_cost = 0, all_network_cost = 0;
    for (std::size_t i = 0; i < cluster.node_ids.size(); i++) {
      int node_id = cluster.node_ids[i];
      node_info &node = nodes_info_[node_id];
      all_storage += node.storage;
      all_bandwidth += node.bandwidth;
      all_storage_cost += node.storage_cost;
      all_network_cost += node.network_cost;
    }
    storage_cost = all_storage_cost / all_storage;
    network_cost = all_network_cost / all_bandwidth;
  }

  void compute_avg_cost_for_each_node_and_cluster(
      double &node_avg_storage_cost, double &node_avg_network_cost,
      double &cluster_avg_storage_cost, double &cluster_avg_network_cost) {
    for (auto &node : nodes_info_) {
      double storage_cost = node.storage_cost / node.storage;
      double network_cost = node.network_cost / node.bandwidth;
      node_avg_storage_cost += storage_cost;
      node_avg_network_cost += network_cost;
    }
    node_avg_storage_cost /= static_cast<double>(nodes_info_.size());
    node_avg_network_cost /= static_cast<double>(nodes_info_.size());

    for (auto &cluster : clusters_info_) {
      double storage_cost = 0, network_cost = 0;
      compute_total_cost_for_cluster(cluster, storage_cost, network_cost);
      cluster_avg_storage_cost += storage_cost;
      cluster_avg_network_cost += network_cost;
    }
    cluster_avg_storage_cost /= static_cast<double>(clusters_info_.size());
    cluster_avg_network_cost /= static_cast<double>(clusters_info_.size());
  }

  void insert_stripes(placement_strategy ps) {
    assert(conf_.num_stripes > 0);
    for (int i = 0; i < conf_.num_stripes; i++) {
      int stripe_id = next_stripe_id_++;

      stripes_info_[stripe_id] = {
          .stripe_id = stripe_id, .K = conf_.K, .L = conf_.L, .G = conf_.G};
      stripe_info &stripe = stripes_info_[stripe_id];

      if (ps == placement_strategy::random) {
        placement_random(stripe_id, stripe.node_ids);
      } else if (ps == placement_strategy::ECWIDE) {
        placement_ECWIDE(stripe_id, stripe.node_ids);
      } else if (ps == placement_strategy::ICPP23_1) {
        placement_ICPP23_1(stripe_id, stripe.node_ids);
      } else if (ps == placement_strategy::ICPP23_2) {
        placement_ICPP23_2(stripe_id, stripe.node_ids);
      } else {
        assert(false);
      }

      for (auto node_id : stripe.node_ids) {
        nodes_info_[node_id].network_cost += 1;
        nodes_info_[node_id].storage_cost += 1;
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
    }
  }

  bool check_data_loss(double failure_rate) {
    int num_failed_nodes = static_cast<double>(conf_.num_nodes) * failure_rate;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, nodes_info_.size() - 1);
    std::unordered_set<int> failed_node_ids;
    for (int i = 0; i < num_failed_nodes; i++) {
      int node_id;
      do {
        node_id = nodes_info_[dis(gen)].node_id;
      } while (failed_node_ids.contains(node_id));
      assert(failed_node_ids.contains(node_id) == false);
      failed_node_ids.insert(node_id);
    }
    assert(failed_node_ids.size() == num_failed_nodes);

    // 假设任意g + 2个块损坏，数据丢失
    int num_failed_blocks_to_data_loss = conf_.G + 2;
    for (auto stripe : stripes_info_) {
      int num_node_in_failure_range = 0;

      for (auto node_id : stripe.second.node_ids) {
        if (failed_node_ids.contains(node_id)) {
          num_node_in_failure_range++;
        }
      }

      if (num_node_in_failure_range >= num_failed_blocks_to_data_loss) {
        return true;
      }
    }

    return false;
  }

  void show_partition(partition_strategy ps) {
    std::vector<std::vector<int>> partition_plan;
    std::string ps_str;
    if (ps == partition_strategy::random) {
      partition_plan = partition_random();
      ps_str = "partition_strategy::random";
    } else if (ps == partition_strategy::ECWIDE) {
      partition_plan = partition_ECWIDE();
      ps_str = "partition_strategy::ECWIDE";
    } else if (ps == partition_strategy::ICPP23) {
      partition_plan = partition_ICPP23();
      ps_str = "partition_strategy::ICPP23";
    } else {
      assert(false);
    }
    std::cout << std::format("partition strategy {}\n", ps_str);
    print2DVector(partition_plan);
  }

  void show_nodes_info() {
    for (auto node : nodes_info_) {
      node.print_info();
    }
  }

  void show_placement(placement_strategy ps) {
    std::vector<int> selected_nodes;

    std::string ps_str;
    int stripe_id = 0;
    stripe_info temp_stripe{
        .stripe_id = stripe_id, .K = conf_.K, .L = conf_.L, .G = conf_.G};
    stripes_info_[stripe_id] = temp_stripe;
    if (ps == placement_strategy::random) {
      placement_random(stripe_id, stripes_info_[stripe_id].node_ids);
      ps_str = "placement_strategy::random";
    } else if (ps == placement_strategy::ECWIDE) {
      placement_ECWIDE(stripe_id, stripes_info_[stripe_id].node_ids);
      ps_str = "placement_strategy::ECWIDE";
    } else if (ps == placement_strategy::ICPP23_1) {
      placement_ICPP23_1(stripe_id, stripes_info_[stripe_id].node_ids);
      ps_str = "placement_strategy::ICPP23_1";
    } else if (ps == placement_strategy::ICPP23_2) {
      placement_ICPP23_2(stripe_id, stripes_info_[stripe_id].node_ids);
      ps_str = "placement_strategy::ICPP23_2";
    } else {
      assert(false);
    }

    std::cout << std::format("placement strategy {}\n", ps_str);
    for (auto node_id : stripes_info_[stripe_id].node_ids) {
      nodes_info_[node_id].print_info();
    }

    stripes_info_.erase(stripe_id);
  }

  void show_load_balance() {
    {
      double max_storage_cost = std::numeric_limits<double>::min(),
             avg_storage_cost = 0;
      double max_network_cost = std::numeric_limits<double>::min(),
             avg_network_cost = 0;
      double storage_bias = 0;
      double network_bias = 0;
      for (auto &p : nodes_info_) {
        double storage_cost = p.storage_cost / p.storage;
        double network_cost = p.network_cost / p.bandwidth;
        max_storage_cost = std::max(max_storage_cost, storage_cost);
        max_network_cost = std::max(max_network_cost, network_cost);
        avg_storage_cost += storage_cost;
        avg_network_cost += network_cost;
      }
      avg_storage_cost /= static_cast<double>(nodes_info_.size());
      avg_network_cost /= static_cast<double>(nodes_info_.size());
      storage_bias = (max_storage_cost - avg_storage_cost) / avg_storage_cost;
      network_bias = (max_network_cost - avg_network_cost) / avg_network_cost;
      std::cout << std::format("节点级别的load balance测试：存储{}，网络{}。\n",
                               storage_bias, network_bias);
    }

    {
      double max_storage_cost = std::numeric_limits<double>::min(),
             avg_storage_cost = 0;
      double max_network_cost = std::numeric_limits<double>::min(),
             avg_network_cost = 0;
      double storage_bias = 0;
      double network_bias = 0;
      for (auto &p : clusters_info_) {
        double storage_cost = 0, network_cost = 0;
        compute_total_cost_for_cluster(p, storage_cost, network_cost);
        max_storage_cost = std::max(max_storage_cost, storage_cost);
        max_network_cost = std::max(max_network_cost, network_cost);
        avg_storage_cost += storage_cost;
        avg_network_cost += network_cost;
      }
      avg_storage_cost /= static_cast<double>(clusters_info_.size());
      avg_network_cost /= static_cast<double>(clusters_info_.size());
      storage_bias = (max_storage_cost - avg_storage_cost) / avg_storage_cost;
      network_bias = (max_network_cost - avg_network_cost) / avg_network_cost;
      std::cout << std::format("集群级别的load balance测试：存储{}，网络{}。\n",
                               storage_bias, network_bias);
    }
  }

  static void show_data_loss_probability(placement_config conf, int test_times,
                                         placement_strategy ps,
                                         double failure_rate = 0.01) {
    int data_loss_times = 0;
    for (int i = 0; i < test_times; i++) {
      data_placement data_loss(conf);
      data_loss.show_partition(partition_strategy::ICPP23);
      data_loss.insert_stripes(ps);
      if (data_loss.check_data_loss(failure_rate)) {
        data_loss_times++;
      }
      std::cout << std::format("第{}次实验，一共失败了{}次:\n", i,
                               data_loss_times);
    }

    double loss_probability =
        static_cast<double>(data_loss_times) / static_cast<double>(test_times);
    std::cout << std::format("数据丢失的概率为{}\n", loss_probability);
  }

  void show_repair_load_distribution(int node_id) {
    node_info &node = nodes_info_[node_id];

    std::vector<int> failed_stripe_ids;
    for (auto stripe_id : nodes_info_[node_id].stripe_ids) {
      failed_stripe_ids.push_back(stripe_id);
    }

    std::map<int, int> repair_load_distribution_node_level;
    std::map<int, int> repair_load_distribution_cluster_level;
    for (auto stripe_id : failed_stripe_ids) {
      // 找到条带内的哪一个block损坏了
      int failed_block_index = -1;
      for (std::size_t i = 0; i < stripes_info_[stripe_id].node_ids.size();
           i++) {
        if (stripes_info_[stripe_id].node_ids[i] == node_id) {
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
      for (auto vec : blocks_to_read_in_each_cluster) {
        for (auto pr : vec) {
          repair_load_distribution_node_level[pr.first]++;
        }
      }
      for (auto cluster_id : repair_span_cluster) {
        if (cluster_id != nodes_info_[node_id].cluster_id) {
          repair_load_distribution_cluster_level[cluster_id] += 1;
        }
      }
    }

    int sum = 0;
    for (auto pr : repair_load_distribution_cluster_level) {
      sum += pr.second;
    }
    std::cout << std::format(
        "{:4}号节点失效时：修复IO会分散到{}个集群上，一共{}个跨集群IO\n",
        node_id, repair_load_distribution_cluster_level.size(), sum);
    for (auto pr : repair_load_distribution_cluster_level) {
      std::cout << std::format("在{}号集群上产生{}跨集群IO\n", pr.first,
                               pr.second);
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

        node.network_cost += 1;
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

            node.network_cost += 1;
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

            node.network_cost += 1;
          }
        }
        if (cluster_id == main_cluster_id) {
          assert(blocks_to_read_in_cur_cluster.size() > 0);
        }
        blocks_to_read_in_each_cluster.push_back(blocks_to_read_in_cur_cluster);
      }
    }
  }

  int next_stripe_id_{0};

  placement_config conf_;

  std::vector<cluster_info> clusters_info_;
  std::vector<node_info> nodes_info_;
  std::unordered_map<int, stripe_info> stripes_info_;
};

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

      assert(false);
    } else {
      assert(false);
    }
  }

  config conf_;
};

int main() {
  // for (int num_nodes = 100; num_nodes <= 10000; num_nodes += 100) {
  //   double failure_rate = 0.01;
  //   loss_probability_replication<replication_schema::random> lpr_random(
  //       {.failure_rate = failure_rate,
  //        .num_nodes = num_nodes,
  //        .replication_factor = 3,
  //        .blocks_per_node = 10000,
  //        .scatter_width = -1});
  //   loss_probability_replication<replication_schema::copyset> lpr_copyset(
  //       {.failure_rate = failure_rate,
  //        .num_nodes = num_nodes,
  //        .replication_factor = 3,
  //        .blocks_per_node = 10000,
  //        .scatter_width = 200});
  //   std::cout << std::format(
  //       "系统共有{:6}个节点，有{:.2f}"
  //       "的节点损坏，通过公式计算数据丢失概率，随机模式为{}，copyset模式为{}\n",
  //       num_nodes, failure_rate, lpr_random.by_equation(),
  //       lpr_copyset.by_equation());
  // }

  // data_placement::test_case_partition();
  // data_placement::test_case_placement();
  // data_placement::test_case_load_balance();
  // data_placement::test_case_data_loss_probability();
  data_placement::test_case_repair_load_distribution();

  return 0;
}