#include "zipf.h"
#include <cassert>
#include <cmath>
#include <format>
#include <iostream>
#include <map>

int main() {
  std::map<int, int> records;

  int num_reads = 0;
  int num_writes = 0;

  int num_write_iterations = 31;
  int num_read_iterations = 12;

  int num_ticks = num_read_iterations * num_write_iterations;
  int next_id = 0;
  int read_turn = 0, write_turn = 0;
  for (int i = 1; i <= num_ticks; i++) {
    if (i % num_read_iterations == 0) {
      // write
      for (int j = 1; j <= 100 * (write_turn + 1); j++) {
        records[next_id] = 1;
        next_id++;
        num_writes++;
      }
      write_turn++;
    }
    if (i % num_write_iterations == 0) {
      // read
      std::default_random_engine generator;
      zipfian_int_distribution<int> zipf_dis(0, next_id - 1, 0.9);

      for (int j = 1; j <= 232 * std::pow(2, read_turn); j++) {
        int idx = zipf_dis(generator);
        idx = next_id - 1 - idx;
        assert(records.find(idx) != records.end());
        records[idx]++;
        num_reads++;
      }
      read_turn++;
    }
  }

  std::cout << "Records in unordered_map:" << std::endl;
  for (const auto &pair : records) {
    std::cout << "Key: " << pair.first << ", Value: " << pair.second
              << std::endl;
  }
  std::cout << std::format("num_writes: {}, num_reads: {}, next_id: {}\n",
                           num_writes, num_reads, next_id);

  return 0;
}