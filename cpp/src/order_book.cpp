#include "mmb/order_book.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mmb {

template <class Book, class Cross>
void OrderBook::match(Book& book, uint64_t id, int32_t owner, Side side, int64_t& qty,
                      Cross crosses, std::vector<Fill>& out) {
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
            if (maker.qty == 0) {
                index_.erase(maker.id);
                level.pop_front();
            }
        }
        if (level.empty()) book.erase(lvl);
    }
}

std::vector<Fill> OrderBook::submit_impl(uint64_t id, int32_t owner, Side side,
                                         std::optional<int64_t> price, int64_t qty, TIF tif) {
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

std::vector<Fill> OrderBook::submit_limit(uint64_t id, int32_t owner, Side side, int64_t price,
                                          int64_t qty, TIF tif) {
    return submit_impl(id, owner, side, price, qty, tif);
}

std::vector<Fill> OrderBook::submit_market(uint64_t id, int32_t owner, Side side, int64_t qty) {
    return submit_impl(id, owner, side, std::nullopt, qty, TIF::IOC);
}

bool OrderBook::cancel(uint64_t id) {
    auto it = index_.find(id);
    if (it == index_.end()) return false;
    const Loc loc = it->second;
    index_.erase(it);
    if (loc.side == Side::Buy) {
        auto lvl = bids_.find(loc.price);
        lvl->second.erase(loc.it);
        if (lvl->second.empty()) bids_.erase(lvl);
    } else {
        auto lvl = asks_.find(loc.price);
        lvl->second.erase(loc.it);
        if (lvl->second.empty()) asks_.erase(lvl);
    }
    return true;
}

std::size_t OrderBook::cancel_owner(int32_t owner) {
    std::vector<uint64_t> ids;
    for (const auto& [oid, loc] : index_)
        if (loc.it->owner == owner) ids.push_back(oid);
    for (auto oid : ids) cancel(oid);
    return ids.size();
}

std::optional<int64_t> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<int64_t> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

int64_t OrderBook::volume_at(Side side, int64_t price) const {
    auto sum = [](const Level& l) {
        int64_t s = 0;
        for (const auto& o : l) s += o.qty;
        return s;
    };
    if (side == Side::Buy) {
        auto it = bids_.find(price);
        return it == bids_.end() ? 0 : sum(it->second);
    }
    auto it = asks_.find(price);
    return it == asks_.end() ? 0 : sum(it->second);
}

std::vector<std::pair<int64_t, int64_t>> OrderBook::depth(Side side, std::size_t levels) const {
    std::vector<std::pair<int64_t, int64_t>> out;
    auto collect = [&](const auto& book) {
        for (const auto& [p, level] : book) {
            if (out.size() >= levels) break;
            int64_t s = 0;
            for (const auto& o : level) s += o.qty;
            out.emplace_back(p, s);
        }
    };
    if (side == Side::Buy) collect(bids_);
    else collect(asks_);
    return out;
}

void OrderBook::clear() {
    bids_.clear();
    asks_.clear();
    index_.clear();
}

void Ledger::apply(const Fill& f) {
    const int64_t notional = f.price * f.qty;
    const int64_t sgn = f.taker_side == Side::Buy ? 1 : -1;
    pos_[f.taker_owner] += sgn * f.qty;
    cash_[f.taker_owner] -= sgn * notional;
    pos_[f.maker_owner] -= sgn * f.qty;
    cash_[f.maker_owner] += sgn * notional;
}

int64_t Ledger::position(int32_t owner) const {
    auto it = pos_.find(owner);
    return it == pos_.end() ? 0 : it->second;
}

int64_t Ledger::cash(int32_t owner) const {
    auto it = cash_.find(owner);
    return it == cash_.end() ? 0 : it->second;
}

ReplayStats replay(OrderBook& book, const std::vector<int8_t>& op, const std::vector<uint64_t>& id,
                   const std::vector<int32_t>& owner, const std::vector<int8_t>& side,
                   const std::vector<int64_t>& price, const std::vector<int64_t>& qty) {
    const std::size_t n = op.size();
    if (id.size() != n || owner.size() != n || side.size() != n || price.size() != n ||
        qty.size() != n)
        throw std::invalid_argument("replay: arrays must have equal length");
    ReplayStats st;
    // FNV-style mixing over (maker_id, price, qty) of every fill, in order.
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };
    std::vector<Fill> fills;
    for (std::size_t i = 0; i < n; ++i) {
        const Side s = side[i] == 0 ? Side::Buy : Side::Sell;
        switch (op[i]) {
            case 0: fills = book.submit_limit(id[i], owner[i], s, price[i], qty[i], TIF::GTC); break;
            case 1: fills = book.submit_limit(id[i], owner[i], s, price[i], qty[i], TIF::IOC); break;
            case 2: fills = book.submit_market(id[i], owner[i], s, qty[i]); break;
            case 3:
                st.cancels_ok += book.cancel(id[i]) ? 1 : 0;
                continue;
            default: throw std::invalid_argument("replay: bad op code");
        }
        for (const auto& f : fills) {
            ++st.fills;
            st.volume += f.qty;
            mix(f.maker_id);
            mix(static_cast<uint64_t>(f.price));
            mix(static_cast<uint64_t>(f.qty));
        }
    }
    st.checksum = static_cast<int64_t>(h & 0x7fffffffffffffffULL);
    return st;
}

}  // namespace mmb
