// Baseline (v1) order book kept for the native benchmark only:
// std::map of price levels, std::list of orders per level, std::unordered_map index.
// Same semantics as mmb::OrderBook; used to measure the effect of the flat
// layout in the production engine.
#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "mmb/order_book.hpp"

namespace mmb::v1 {

class MapOrderBook {
public:
    std::vector<Fill> submit_limit(uint64_t id, int32_t owner, Side side, int64_t price, int64_t qty, TIF tif) {
        return submit_impl(id, owner, side, price, qty, tif);
    }
    std::vector<Fill> submit_market(uint64_t id, int32_t owner, Side side, int64_t qty) {
        return submit_impl(id, owner, side, std::nullopt, qty, TIF::IOC);
    }
    bool cancel(uint64_t id, int32_t owner) {
        auto it = index_.find(id);
        if (it == index_.end()) return false;
        const Loc loc = it->second;
        if (loc.it->owner != owner) return false;
        index_.erase(it);
        if (loc.side == Side::Buy) erase_from(bids_, loc);
        else erase_from(asks_, loc);
        return true;
    }

private:
    struct Order { uint64_t id; int32_t owner; int64_t qty; };
    using Level = std::list<Order>;
    struct Loc { Side side; int64_t price; Level::iterator it; };

    template <class Book>
    void erase_from(Book& book, const Loc& loc) {
        auto lvl = book.find(loc.price);
        lvl->second.erase(loc.it);
        if (lvl->second.empty()) book.erase(lvl);
    }

    template <class Book, class Cross>
    void match(Book& book, uint64_t id, int32_t owner, Side side, int64_t& qty, Cross crosses,
               std::vector<Fill>& out) {
        while (qty > 0 && !book.empty()) {
            auto lvl = book.begin();
            if (!crosses(lvl->first)) break;
            Level& level = lvl->second;
            while (qty > 0 && !level.empty()) {
                Order& maker = level.front();
                const int64_t q = std::min(qty, maker.qty);
                out.push_back(Fill{id, maker.id, owner, maker.owner, side, lvl->first, q});
                qty -= q;
                maker.qty -= q;
                if (maker.qty == 0) { index_.erase(maker.id); level.pop_front(); }
            }
            if (level.empty()) book.erase(lvl);
        }
    }

    std::vector<Fill> submit_impl(uint64_t id, int32_t owner, Side side, std::optional<int64_t> price,
                                  int64_t qty, TIF tif) {
        if (qty <= 0) throw std::invalid_argument("qty must be positive");
        if (index_.count(id)) throw std::invalid_argument("duplicate live order id");
        std::vector<Fill> fills;
        if (side == Side::Buy) {
            const int64_t lim = price ? *price : std::numeric_limits<int64_t>::max();
            match(asks_, id, owner, side, qty, [lim](int64_t p) { return p <= lim; }, fills);
            if (qty > 0 && tif == TIF::GTC && price) {
                Level& level = bids_[*price];
                level.push_back(Order{id, owner, qty});
                index_.emplace(id, Loc{side, *price, std::prev(level.end())});
            }
        } else {
            const int64_t lim = price ? *price : std::numeric_limits<int64_t>::min();
            match(bids_, id, owner, side, qty, [lim](int64_t p) { return p >= lim; }, fills);
            if (qty > 0 && tif == TIF::GTC && price) {
                Level& level = asks_[*price];
                level.push_back(Order{id, owner, qty});
                index_.emplace(id, Loc{side, *price, std::prev(level.end())});
            }
        }
        return fills;
    }

    std::map<int64_t, Level, std::greater<int64_t>> bids_;
    std::map<int64_t, Level, std::less<int64_t>> asks_;
    std::unordered_map<uint64_t, Loc> index_;
};

}  // namespace mmb::v1
