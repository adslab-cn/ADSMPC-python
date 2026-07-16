#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "emp-sh2pc/emp-sh2pc.h"
#include "protocol/gc_topk.h"

namespace {

std::vector<uint64_t> read_values(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open input file: " + path);
  }
  std::vector<uint64_t> values;
  std::string token;
  while (in >> token) {
    values.push_back(static_cast<uint64_t>(std::stoull(token)));
  }
  if (values.empty()) {
    throw std::runtime_error("input file is empty");
  }
  return values;
}

void write_ids(const std::string& path, const std::vector<uint64_t>& ids) {
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }
  for (auto id : ids) {
    out << id << "\n";
  }
}

std::vector<uint64_t> public_min_distance_topk(
    int party,
    const std::vector<uint64_t>& score_share,
    size_t k,
    size_t value_bits,
    size_t id_bits,
    size_t bin_count) {
  const size_t n = score_share.size();
  const size_t padded_n = ((n + k - 1) / k) * k;
  std::vector<emp::Integer> scores(padded_n);
  std::vector<emp::Integer> ids(padded_n);

  for (size_t i = 0; i < n; ++i) {
    const uint64_t alice_score_share = party == emp::ALICE ? score_share[i] : 0;
    const uint64_t bob_score_share = party == emp::BOB ? score_share[i] : 0;
    emp::Integer alice_score(static_cast<int>(value_bits), alice_score_share, emp::ALICE);
    emp::Integer bob_score(static_cast<int>(value_bits), bob_score_share, emp::BOB);
    scores[i] = alice_score + bob_score;

    const uint64_t id_share = party == emp::ALICE ? i : 0;
    emp::Integer alice_id(static_cast<int>(id_bits), id_share, emp::ALICE);
    emp::Integer bob_id(static_cast<int>(id_bits), 0, emp::BOB);
    ids[i] = alice_id + bob_id;
  }
  for (size_t i = n; i < padded_n; ++i) {
    const uint64_t max_distance = (uint64_t{1} << (value_bits - 1)) - 1;
    scores[i] = emp::Integer(static_cast<int>(value_bits), max_distance, emp::PUBLIC);
    ids[i] = emp::Integer(static_cast<int>(id_bits), 0, emp::PUBLIC);
  }

  std::vector<emp::Integer> top_scores(k);
  std::vector<emp::Integer> top_ids(k);
  if (bin_count > 0 && bin_count < padded_n) {
    panther::gc::Approximate_topk(
        scores.data(),
        ids.data(),
        static_cast<int>(padded_n),
        static_cast<int>(k),
        static_cast<int>(bin_count),
        top_scores.data(),
        top_ids.data());
  } else {
    panther::gc::BitonicTopk(scores.data(), ids.data(), static_cast<int>(padded_n), static_cast<int>(k), true);
    for (size_t i = 0; i < k; ++i) {
      top_ids[i] = ids[i];
    }
  }
  std::vector<uint64_t> out(k);
  for (size_t i = 0; i < k; ++i) {
    out[i] = top_ids[i].reveal<uint64_t>(emp::BOB);
    if (party == emp::BOB && out[i] >= n) {
      throw std::runtime_error("GC top-k returned a padded id");
    }
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8 && argc != 9) {
    std::cerr << "usage: panther_gc_topk_cli <party:1|2> <port> <k> "
                 "<value_bits> <id_bits> <input_file> <output_file> [bin_count]\n";
    return 2;
  }
  try {
    const int party = std::stoi(argv[1]);
    const int port = std::stoi(argv[2]);
    const size_t k = static_cast<size_t>(std::stoull(argv[3]));
    const size_t value_bits = static_cast<size_t>(std::stoull(argv[4]));
    const size_t id_bits = static_cast<size_t>(std::stoull(argv[5]));
    const std::string input_file = argv[6];
    const std::string output_file = argv[7];
    const size_t bin_count = argc == 9 ? static_cast<size_t>(std::stoull(argv[8])) : 0;
    if (party != emp::ALICE && party != emp::BOB) {
      throw std::runtime_error("party must be 1 or 2");
    }
    if (k == 0 || value_bits == 0 || value_bits > 31 || id_bits == 0 || id_bits > 64) {
      throw std::runtime_error("invalid k/value_bits/id_bits");
    }

    auto values = read_values(input_file);
    if (k > values.size()) {
      throw std::runtime_error("k is larger than input length");
    }

    emp::NetIO io(party == emp::ALICE ? nullptr : "127.0.0.1", port);
    emp::setup_semi_honest(&io, party);
    auto ids = public_min_distance_topk(party, values, k, value_bits, id_bits, bin_count);
    emp::finalize_semi_honest();
    write_ids(output_file, ids);
    std::cerr << "communication_bytes=" << io.counter << "\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "panther_gc_topk_cli error: " << exc.what() << "\n";
    return 1;
  }
}
