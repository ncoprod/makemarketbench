"""Interleave old/new native books in one process on the identical saved stream.

Builds the old source from git into a separate namespace, avoiding an additional
vendored book. Keep every sample: concurrent laptop workloads can dominate wall
time. Thread CPU time is diagnostic only; headline events/s use wall time.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np

from makemarket.provenance import file_sha256, receipt

HARNESS = r'''
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ctime>
#define mmb old_mmb
#include "old_order_book.hpp"
#undef mmb
#include "mmb/order_book.hpp"
double cpu_seconds() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
int main(int argc, char** argv) {
    std::ifstream in(argv[1], std::ios::binary);
    int64_t n;
    in.read(reinterpret_cast<char*>(&n), 8);
    std::vector<int64_t> cols(6 * n);
    in.read(reinterpret_cast<char*>(cols.data()), 48 * n);
    if (!in) return 2;
    std::vector<int8_t> op(n), side(n);
    std::vector<uint64_t> id(n);
    std::vector<int32_t> owner(n);
    std::vector<int64_t> price(n), qty(n);
    for (int64_t i = 0; i < n; ++i) {
        op[i] = cols[i]; id[i] = cols[n+i]; owner[i] = cols[2*n+i];
        side[i] = cols[3*n+i]; price[i] = cols[4*n+i]; qty[i] = cols[5*n+i];
    }
    for (int pair = 0; pair < std::atoi(argv[2]); ++pair) {
        for (int j = 0; j < 2; ++j) {
            const bool old = (pair + j) % 2 == 0;
            old_mmb::OrderBook before;
            mmb::OrderBook after;
            double c0 = cpu_seconds();
            auto t0 = std::chrono::steady_clock::now();
            int64_t checksum;
            if (old) checksum = old_mmb::replay(before,op,id,owner,side,price,qty).checksum;
            else checksum = mmb::replay(after,op,id,owner,side,price,qty).checksum;
            double wall = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
            double cpu = cpu_seconds()-c0;
            std::printf("{\"pair\":%d,\"old\":%s,\"wall_seconds\":%.9f,\"cpu_seconds\":%.9f,\"checksum\":%lld}\n",
                        pair,old?"true":"false",wall,cpu,static_cast<long long>(checksum));
        }
    }
}
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", default="deb9b65")
    ap.add_argument("--stream", default="build/bench_stream.bin")
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--out", default="results/engine_comparison.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = root / "build/engine_comparison"
    build.mkdir(parents=True, exist_ok=True)
    # An executable called "compare" here shadows C++20's <compare> header.
    (build / "compare").unlink(missing_ok=True)
    for path, dest in [("cpp/include/mmb/order_book.hpp", "old_order_book.hpp"),
                       ("cpp/src/order_book.cpp", "old_order_book.cpp")]:
        content = subprocess.check_output(["git", "show", f"{args.before}:{path}"], text=True)
        if path.endswith(".cpp"):
            content = content.replace('"mmb/order_book.hpp"', '"old_order_book.hpp"')
        (build / dest).write_text(content)
    (build / "compare.cpp").write_text(HARNESS)
    common = ["c++", "-O3", "-DNDEBUG", "-std=c++20", "-I", str(build), "-I", str(root / "cpp/include")]
    commands = [common + ["-Dmmb=old_mmb", "-c", str(build / "old_order_book.cpp"), "-o", str(build / "old.o")],
                common + [str(build / "compare.cpp"), str(root / "cpp/src/order_book.cpp"),
                          str(build / "old.o"), "-o", str(build / "bench_comparison")]]
    record = receipt(vars(args))
    record["source_sha256"] = {f: file_sha256(root / f)
                               for f in ("cpp/include/mmb/order_book.hpp", "cpp/src/order_book.cpp")}
    record["compiler"] = subprocess.check_output(["c++", "--version"], text=True).strip()
    record["compile_commands"] = commands
    for command in commands:
        subprocess.run(command, check=True)
    command = [str(build / "bench_comparison"), args.stream, str(args.pairs)]
    samples = [json.loads(line) for line in subprocess.check_output(command, text=True).splitlines()]
    assert len({s["checksum"] for s in samples}) == 1
    n = int.from_bytes(Path(args.stream).read_bytes()[:8], "little")
    summary = {}
    for old, label in ((True, "before"), (False, "after")):
        subset = [s for s in samples if s["old"] == old]
        summary[label] = {"median_events_per_second": n / statistics.median(s["wall_seconds"] for s in subset),
                          "median_cpu_events_per_second": n / statistics.median(s["cpu_seconds"] for s in subset)}
    record.update(before_commit=subprocess.check_output(["git", "rev-parse", args.before], text=True).strip(),
                  events=n, stream_sha256=file_sha256(args.stream), samples=samples, summary=summary,
                  checksum=samples[0]["checksum"])
    before = np.array([s["wall_seconds"] for s in samples if s["old"]])
    after = np.array([s["wall_seconds"] for s in samples if not s["old"]])
    ratios = before / after
    rng = np.random.default_rng(0)
    boot = np.median(ratios[rng.integers(0, len(ratios), (10000, len(ratios)))], axis=1)
    record["paired_median_speedup"] = float(np.median(ratios))
    record["paired_median_speedup_ci_95"] = np.quantile(boot, [0.025, 0.975]).tolist()
    record["paired_median_speedup_ci_bonferroni_3"] = np.quantile(boot, [0.05 / 6, 1 - 0.05 / 6]).tolist()
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
