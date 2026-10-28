// MakeMarketBench limit order book: price-time priority matching engine.
//
// Semantics (mirrored exactly by makemarket.engine.pybook):
//   * Prices and quantities are integers (prices in ticks).
//   * An incoming order matches against the opposite side while the best
//     opposite price is marketable (ask <= limit for buys, bid >= limit for
//     sells). Within a level, makers are filled in arrival (FIFO) order.
//   * Every fill executes at the maker's resting price.
//   * The unfilled remainder of a GTC order rests in the book; the remainder
//     of an IOC or market order is discarded.
//   * No self-trade prevention. Order ids must be unique among live orders
//     (id 2^64-1 is reserved).
//   * Only the owner of an order can cancel it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mmb {

enum class Side : int8_t { Buy = 0, Sell = 1 };
enum class TIF : int8_t { GTC = 0, IOC = 1 };

struct Fill {
    uint64_t taker_id;
    uint64_t maker_id;
    int32_t taker_owner;
    int32_t maker_owner;
    Side taker_side;
    int64_t price;
    int64_t qty;
};

struct ReplayStats;
class OrderBook;
ReplayStats replay(OrderBook& book, const std::vector<int8_t>& op, const std::vector<uint64_t>& id,
                   const std::vector<int32_t>& owner, const std::vector<int8_t>& side,
                   const std::vector<int64_t>& price, const std::vector<int64_t>& qty);

class OrderBook {
public:
    // Submit a limit order. Returns the fills it generated (taker = this order).
    std::vector<Fill> submit_limit(uint64_t id, int32_t owner, Side side, int64_t price,
                                   int64_t qty, TIF tif);
    // Submit a market order (IOC, no price limit).
    std::vector<Fill> submit_market(uint64_t id, int32_t owner, Side side, int64_t qty);
    // Cancel a resting order on behalf of `owner`. Returns false if the id is not
    // live or belongs to another owner (agents cannot cancel each other's orders).
    bool cancel(uint64_t id, int32_t owner);
    // Cancel all resting orders of an owner. Returns number cancelled.
    std::size_t cancel_owner(int32_t owner);

    std::optional<int64_t> best_bid() const;
    std::optional<int64_t> best_ask() const;
    int64_t volume_at(Side side, int64_t price) const;
    // Aggregated depth: (price, total qty) from best to worse, up to `levels`.
    std::vector<std::pair<int64_t, int64_t>> depth(Side side, std::size_t levels) const;
    std::size_t order_count() const { return index_.size(); }
    bool has_order(uint64_t id) const { return index_.find(id) != kNil; }
    void clear();

private:
    // Orders live in a pool and form an intrusive FIFO list per price level, so
    // resting an order does not allocate once the pool has grown. Price levels
    // are kept in flat vectors sorted so that the best price is at the back:
    // matching pops from the back, and new levels near the touch are cheap to insert.
    static constexpr int32_t kNil = -1;
    struct Node {
        uint64_t id;
        int64_t qty;
        int64_t price;
        int32_t owner;
        int32_t prev;
        int32_t next;
        Side side;
    };
    struct Level {
        int64_t price;
        int64_t total;
        int32_t head;
        int32_t tail;
    };
    // Open-addressing (linear probing, backward-shift deletion) id -> node map.
    class IdMap {
    public:
        IdMap() { rehash(1024); }
        int32_t find(uint64_t key) const;
        void insert(uint64_t key, int32_t value);
        void erase(uint64_t key);
        std::size_t size() const { return size_; }
        void clear() { std::fill(keys_.begin(), keys_.end(), kEmpty); size_ = 0; }
        template <class F> void for_each(F f) const {
            for (std::size_t i = 0; i < keys_.size(); ++i)
                if (keys_[i] != kEmpty) f(keys_[i], vals_[i]);
        }
        static constexpr uint64_t kEmpty = ~uint64_t{0};

    private:
        static std::size_t hash(uint64_t k) {
            k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
            return static_cast<std::size_t>(k);
        }
        void rehash(std::size_t cap);
        std::vector<uint64_t> keys_;
        std::vector<int32_t> vals_;
        std::size_t mask_ = 0;
        std::size_t size_ = 0;
    };

    int32_t alloc_node(const Node& n);
    void unlink(int32_t idx, std::vector<Level>& levels, std::size_t li);
    void rest(uint64_t id, int32_t owner, Side side, int64_t price, int64_t qty);
    template <bool IsBuy>
    void match(uint64_t id, int32_t owner, Side side, int64_t limit, int64_t& qty, std::vector<Fill>& out);
    void submit_impl(uint64_t id, int32_t owner, Side side, bool has_price, int64_t price, int64_t qty,
                     TIF tif, std::vector<Fill>& out);
    friend ReplayStats replay(OrderBook&, const std::vector<int8_t>&, const std::vector<uint64_t>&,
                              const std::vector<int32_t>&, const std::vector<int8_t>&,
                              const std::vector<int64_t>&, const std::vector<int64_t>&);
    // index of the level with `price` in `levels`, or levels.size() if absent
    static std::size_t find_level(const std::vector<Level>& levels, int64_t price, bool is_bid);

    std::vector<Node> pool_;
    std::vector<int32_t> free_;
    std::vector<Level> bids_;  // ascending price: best bid at back
    std::vector<Level> asks_;  // descending price: best ask at back
    IdMap index_;
};

// Cash/position ledger. Cash is in tick units (price * qty).
class Ledger {
public:
    void apply(const Fill& f);
    int64_t position(int32_t owner) const;
    int64_t cash(int32_t owner) const;
    // Mark-to-value in ticks: cash + position * value.
    int64_t mark(int32_t owner, int64_t value) const { return cash(owner) + position(owner) * value; }
    void clear() { pos_.clear(); cash_.clear(); }

private:
    std::unordered_map<int32_t, int64_t> pos_;
    std::unordered_map<int32_t, int64_t> cash_;
};

// Order book + ledger: every fill is booked to both counterparties.
class Exchange {
public:
    std::vector<Fill> submit_limit(uint64_t id, int32_t owner, Side side, int64_t price,
                                   int64_t qty, TIF tif) {
        auto fills = book.submit_limit(id, owner, side, price, qty, tif);
        for (const auto& f : fills) ledger.apply(f);
        return fills;
    }
    std::vector<Fill> submit_market(uint64_t id, int32_t owner, Side side, int64_t qty) {
        auto fills = book.submit_market(id, owner, side, qty);
        for (const auto& f : fills) ledger.apply(f);
        return fills;
    }
    bool cancel(uint64_t id, int32_t owner) { return book.cancel(id, owner); }
    OrderBook book;
    Ledger ledger;
};

// Replay a stream of operations (used by the benchmarks and parity tests).
// op: 0 = limit GTC, 1 = limit IOC, 2 = market, 3 = cancel (by owner[i]).
struct ReplayStats {
    uint64_t fills = 0;
    int64_t volume = 0;
    uint64_t cancels_ok = 0;
    int64_t checksum = 0;  // order-sensitive hash of fills (for parity checks)
};
ReplayStats replay(OrderBook& book, const std::vector<int8_t>& op, const std::vector<uint64_t>& id,
                   const std::vector<int32_t>& owner, const std::vector<int8_t>& side,
                   const std::vector<int64_t>& price, const std::vector<int64_t>& qty);

}  // namespace mmb
