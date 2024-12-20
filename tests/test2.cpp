#include "../tools.hpp"
#include <algorithm>
#include <cstddef>
#include <fmt/core.h>
#include <iostream>
#include <map>
#include <random>
#include <vector>

static constexpr int K = 4;
static constexpr int L = 2;
static constexpr int G = 2;

// 写入多个条带，但确保0号数据块始终位于0号节点
// 测试当0号节点损坏，有哪些节点参与了修复，参与了几次
void case1() {
  std::vector<int> blocks_permutation{0, 1, 2, 3, 4, 5, 6};

  std::map<int, int> nodes{};
  bool flag{true};
  while (flag && blocks_permutation.front() == 0) {
    // print_nums(blocks_permutation);
    for (std::size_t node = 1; node < K + L + G; node++) {
      auto block = blocks_permutation[node];
      if (block == 1 || block == 2) {
        nodes[node]++;
      }
    }

    flag = std::next_permutation(blocks_permutation.begin(),
                                 blocks_permutation.end());
  }

  for (auto &pr : nodes) {
    std::cout
        << fmt::format(
               "为了修复0号节点上存储的0号数据块，{}号节点参与修复的次数{}",
               pr.first, pr.second)
        << std::endl;
  }

  std::cout << std::endl;
}

// 写入多个条带，但确保6号全局校验块始终位于0号节点
// 测试当0号节点损坏，有哪些节点参与了修复，参与了几次
void case2() {
  std::vector<int> blocks_permutation{6, 0, 1, 2, 3, 4, 5};

  std::map<int, int> nodes{};
  bool flag{true};
  while (flag && blocks_permutation.front() == 6) {
    // print_nums(blocks_permutation);
    for (std::size_t node = 1; node < K + L + G; node++) {
      auto block = blocks_permutation[node];
      if (block == 0 || block == 1 || block == 3 || block == 4) {
        nodes[node]++;
      }
    }

    flag = std::next_permutation(blocks_permutation.begin(),
                                 blocks_permutation.end());
  }

  for (auto &pr : nodes) {
    std::cout
        << fmt::format(
               "为了修复0号节点上存储的6号全局校验块，{}号节点参与修复的次数{}",
               pr.first, pr.second)
        << std::endl;
  }

  std::cout << std::endl;
}

void case3() {
  std::vector<int> blocks_permutation{0, 1, 2, 3, 4, 5, 6};

  // std::vector<std::vector<int>> permutations{};
  // bool flag{true};
  // while (flag) {
  //   permutations.push_back(blocks_permutation);

  //   flag = std::next_permutation(blocks_permutation.begin(),
  //                                blocks_permutation.end());
  // }

  // std::cout << fmt::format("一共有{}种排列情况", permutations.size())
  //           << std::endl;

  int count = 100000;
  std::vector<std::vector<int>> nodes(K + L + G,
                                      std::vector<int>(K + L + G, 0));
  std::random_device rd{};
  std::mt19937 g(rd());
  for (int i = 0; i < count; i++) {
    std::shuffle(blocks_permutation.begin(), blocks_permutation.end(), g);
    for (auto node = 0; node < K + L + G; node++) {
      auto block = blocks_permutation[node];
      nodes[node][block]++;
    }
  }

  for (auto node = 0; node < K + L + G; node++) {
    for (auto block = 0; block < K + L + G; block++) {
      std::cout << fmt::format("节点{}上存储了{}个{}号块", node,
                               nodes[node][block], block)
                << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
}

std::vector<int> request_nodes_to_repair_node(int block_id) {
  std::map<int, std::vector<int>> repair_groups{
      {0, {1, 2}}, {1, {0, 2}}, {2, {0, 1}},       {3, {4, 5}},
      {4, {3, 5}}, {5, {3, 4}}, {6, {0, 1, 3, 4}}, {7, {0, 1, 3, 4}}};
  return repair_groups[block_id];
}

std::vector<int>
block_id_to_node_id(const std::vector<int> &block_ids,
                    const std::vector<int> &blocks_permutation) {
  std::vector<int> node_ids{};
  for (auto block_id : block_ids) {
    for (auto node_id = 0; node_id < K + L + G; node_id++) {
      if (blocks_permutation[node_id] == block_id) {
        node_ids.push_back(node_id);
      }
    }
  }
  return node_ids;
}

void case4() {
  std::vector<int> blocks_permutation{0, 1, 2, 3, 4, 5, 6, 7};
  int count = 50;
  std::random_device rd{};
  std::mt19937 g(rd());
  std::vector<int> edges(K + L + G, 0);
  for (int i = 0; i < count; i++) {
    std::shuffle(blocks_permutation.begin(), blocks_permutation.end(), g);
    int block_id = blocks_permutation[0];
    auto node_ids_to_repair = block_id_to_node_id(
        request_nodes_to_repair_node(block_id), blocks_permutation);

    // print_nums(blocks_permutation);
    // print_nums(node_ids_to_repair);

    for (auto node_id : node_ids_to_repair) {
      edges[node_id]++;
    }
  }

  for (std::size_t i = 0; i < edges.size(); i++) {
    std::cout << fmt::format("0号节点损坏时需要访问{}号节点{}次", i, edges[i])
              << std::endl;
  }

  std::cout << std::endl;
}

int main() {
  // case1();
  // case2();
  // case3();
  case4();

  return 0;
}