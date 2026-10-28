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
//   * No self-trade prevention. Order ids must be unique among live orders.
#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <map>
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
    bool has_order(uint64_t id) const { return index_.count(id) != 0; }
    void clear();

private:
    struct Order {
        uint64_t id;
        int32_t owner;
        int64_t qty;
    };
    using Level = std::list<Order>;
    struct Loc {
        Side side;
        int64_t price;
        Level::iterator it;
    };

    template <class Book, class Cross>
    void match(Book& book, uint64_t id, int32_t owner, Side side, int64_t& qty, Cross crosses,
               std::vector<Fill>& out);
    std::vector<Fill> submit_impl(uint64_t id, int32_t owner, Side side,
                                  std::optional<int64_t> price, int64_t qty, TIF tif);

    std::map<int64_t, Level, std::greater<int64_t>> bids_;
    std::map<int64_t, Level, std::less<int64_t>> asks_;
    std::unordered_map<uint64_t, Loc> index_;
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
