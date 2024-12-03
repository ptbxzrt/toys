#pragma once

#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
#include <utility>

template <typename T> void print_nums(T &nums) {
  for (auto iter = nums.begin(); iter != nums.end(); iter++) {
    std::cout << std::format("{:4} ", *iter);
  }
  std::cout << std::endl;
}

template <typename T> std::pair<double, double> get_avg_and_variance(T &nums) {
  double sum = std::accumulate(nums.begin(), nums.end(), 0.0);
  double avg = sum / static_cast<double>(nums.size());

  // std::cout << std::format("平均值为{}", avg) << std::endl;

  double variance = 0;
  for (auto num : nums) {
    variance += std::pow(static_cast<double>(num) - avg, 2);
  }

  return {avg, std::sqrt(variance / static_cast<double>(nums.size()))};
}

inline double get_combination(int N, int K) {
  double result = 1;
  for (int i = 0; i < K; i++) {
    result *= N;
    N--;
  }
  while (K > 0) {
    result /= K;
    K--;
  }
  return result;
}