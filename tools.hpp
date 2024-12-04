#pragma once

#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

template <typename T> void print_nums(T &nums) {
  for (auto iter = nums.begin(); iter != nums.end(); iter++) {
    std::cout << std::format("{:4} ", *iter);
  }
  std::cout << std::endl;
}

template <typename T> std::pair<double, double> get_avg_and_variance(T &nums) {
  double sum = std::accumulate(nums.begin(), nums.end(), 0.0);
  double avg = sum / static_cast<double>(nums.size());

  double variance = 0;
  for (auto num : nums) {
    variance += std::pow(static_cast<double>(num) - avg, 2);
  }

  return {avg, std::sqrt(variance / static_cast<double>(nums.size()))};
}

inline double get_combination(double N, double K) {
  if (K > N) {
    return 0;
  }
  if (K == 0 || K == N) {
    return 1;
  }

  double result = 1;
  int counter = K;
  for (int i = 0; i < counter; i++) {
    result *= N;
    result /= K;
    N--;
    K--;
  }
  return result;
}

inline void print1DVector(const std::vector<int> &vec) {
  for (const auto &elem : vec) {
    std::cout << std::format("{:4}", elem) << " ";
  }
  std::cout << '\n';
}

inline void print2DVector(const std::vector<std::vector<int>> &vec) {
  for (const auto &row : vec) {
    for (const auto &elem : row) {
      std::cout << std::format("{:4}", elem) << " ";
    }
    std::cout << '\n';
  }
}