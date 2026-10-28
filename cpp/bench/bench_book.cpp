// Native (no Python) throughput benchmark for the order book.
// Usage: bench_book <stream.bin> [repeats]
// stream.bin layout: int64 n, then 6 arrays of n int64: op, id, owner, side, price, qty.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "mmb/order_book.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s stream.bin [repeats]\n", argv[0]);
        return 2;
    }
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 5;
    std::ifstream in(argv[1], std::ios::binary);
    int64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<int64_t> cols(6 * static_cast<std::size_t>(n));
    in.read(reinterpret_cast<char*>(cols.data()), static_cast<std::streamsize>(cols.size() * 8));
    if (!in) {
        std::fprintf(stderr, "failed to read stream\n");
        return 1;
    }
    std::vector<int8_t> op(n), side(n);
    std::vector<uint64_t> id(n);
    std::vector<int32_t> owner(n);
    std::vector<int64_t> price(n), qty(n);
    for (int64_t i = 0; i < n; ++i) {
        op[i] = static_cast<int8_t>(cols[i]);
        id[i] = static_cast<uint64_t>(cols[n + i]);
        owner[i] = static_cast<int32_t>(cols[2 * n + i]);
        side[i] = static_cast<int8_t>(cols[3 * n + i]);
        price[i] = cols[4 * n + i];
        qty[i] = cols[5 * n + i];
    }
    double best = 1e300;
    mmb::ReplayStats st;
    for (int r = 0; r < repeats; ++r) {
        mmb::OrderBook book;
        auto t0 = std::chrono::steady_clock::now();
        st = mmb::replay(book, op, id, owner, side, price, qty);
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    std::printf("{\"events\": %lld, \"best_seconds\": %.6f, \"events_per_sec\": %.1f, "
                "\"fills\": %llu, \"volume\": %lld, \"checksum\": %lld}\n",
                static_cast<long long>(n), best, n / best,
                static_cast<unsigned long long>(st.fills), static_cast<long long>(st.volume),
                static_cast<long long>(st.checksum));
    return 0;
}
