#include "../tools.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fmt/core.h>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static constexpr std::size_t num_nodes = 500;
static constexpr std::uint32_t scatter_width = 10;
static constexpr std::uint32_t replication = 3;
static constexpr std::uint32_t P = scatter_width / (replication - 1) + 1;

int main() {
  std::vector<int> nodes{};
  for (std::size_t i = 0; i < num_nodes; i++) {
    nodes.push_back(i);
  }

  std::random_device rd{};
  std::mt19937 g(rd());

  std::vector<std::vector<int>> copysets{};
  for (int i = 0; i < P; i++) {
    std::shuffle(nodes.begin(), nodes.end(), g);
    // print_nums(nodes);
    for (std::size_t j = 0; j + replication < nodes.size(); j += replication) {
      copysets.push_back(
          std::vector<int>(nodes.begin() + j, nodes.begin() + j + replication));
      // print_nums(copysets.back());
    }
  }

  std::unordered_map<int, std::unordered_set<int>> node_neighbors{};
  for (auto &copyset : copysets) {
    for (std::size_t i = 0; i < copyset.size(); i++) {
      for (std::size_t j = i + 1; j < copyset.size(); j++) {
        node_neighbors[copyset[i]].insert(copyset[j]);
        node_neighbors[copyset[j]].insert(copyset[i]);
      }
    }
  }

  for (auto &neighbor : node_neighbors) {
    std::cout << fmt::format("节点{}的邻居有{}个", neighbor.first,
                             neighbor.second.size())
              << std::endl;
  }

  return 0;
}