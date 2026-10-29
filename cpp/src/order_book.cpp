#include "mmb/order_book.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mmb {

// ------------------------------------------------------------------ IdMap
void OrderBook::IdMap::rehash(std::size_t cap) {
    std::vector<uint64_t> old_keys = std::move(keys_);
    std::vector<int32_t> old_vals = std::move(vals_);
    keys_.assign(cap, kEmpty);
    vals_.assign(cap, kNil);
    mask_ = cap - 1;
    size_ = 0;
    for (std::size_t i = 0; i < old_keys.size(); ++i)
        if (old_keys[i] != kEmpty) insert(old_keys[i], old_vals[i]);
}

int32_t OrderBook::IdMap::find(uint64_t key) const {
    if (key == kEmpty) return kNil;
    for (std::size_t i = hash(key) & mask_;; i = (i + 1) & mask_) {
        if (keys_[i] == key) return vals_[i];
        if (keys_[i] == kEmpty) return kNil;
    }
}

void OrderBook::IdMap::insert(uint64_t key, int32_t value) {
    if ((size_ + 1) * 2 > keys_.size()) rehash(keys_.size() * 2);
    std::size_t i = hash(key) & mask_;
    while (keys_[i] != kEmpty && keys_[i] != key) i = (i + 1) & mask_;
    if (keys_[i] == kEmpty) ++size_;
    keys_[i] = key;
    vals_[i] = value;
}

void OrderBook::IdMap::erase(uint64_t key) {
    std::size_t i = hash(key) & mask_;
    while (keys_[i] != key) {
        if (keys_[i] == kEmpty) return;
        i = (i + 1) & mask_;
    }
    keys_[i] = kEmpty;
    --size_;
    // backward-shift deletion keeps probe sequences intact without tombstones
    for (std::size_t j = (i + 1) & mask_; keys_[j] != kEmpty; j = (j + 1) & mask_) {
        const std::size_t k = hash(keys_[j]) & mask_;
        const bool movable = (j > i) ? (k <= i || k > j) : (k <= i && k > j);
        if (movable) {
            keys_[i] = keys_[j];
            vals_[i] = vals_[j];
            keys_[j] = kEmpty;
            i = j;
        }
    }
}

// -------------------------------------------------------------- OrderBook
std::size_t OrderBook::find_level(const std::vector<Level>& levels, int64_t price, bool is_bid) {
    auto it = is_bid ? std::lower_bound(levels.begin(), levels.end(), price,
                                        [](const Level& l, int64_t p) { return l.price < p; })
                     : std::lower_bound(levels.begin(), levels.end(), price,
                                        [](const Level& l, int64_t p) { return l.price > p; });
    if (it == levels.end() || it->price != price) return levels.size();
    return static_cast<std::size_t>(it - levels.begin());
}

int32_t OrderBook::alloc_node(const Node& n) {
    if (!free_.empty()) {
        const int32_t idx = free_.back();
        free_.pop_back();
        pool_[idx] = n;
        return idx;
    }
    pool_.push_back(n);
    return static_cast<int32_t>(pool_.size() - 1);
}

void OrderBook::rest(uint64_t id, int32_t owner, Side side, int64_t price, int64_t qty) {
    const bool is_bid = side == Side::Buy;
    std::vector<Level>& levels = is_bid ? bids_ : asks_;
    auto it = is_bid ? std::lower_bound(levels.begin(), levels.end(), price,
                                        [](const Level& l, int64_t p) { return l.price < p; })
                     : std::lower_bound(levels.begin(), levels.end(), price,
                                        [](const Level& l, int64_t p) { return l.price > p; });
    if (it == levels.end() || it->price != price) it = levels.insert(it, Level{price, 0, kNil, kNil});
    Level& level = *it;
    const int32_t idx = alloc_node(Node{id, qty, price, owner, level.tail, kNil, side});
    if (level.tail != kNil) pool_[level.tail].next = idx;
    else level.head = idx;
    level.tail = idx;
    level.total += qty;
    index_.insert(id, idx);
}

void OrderBook::unlink(int32_t idx, std::vector<Level>& levels, std::size_t li) {
    Node& n = pool_[idx];
    Level& level = levels[li];
    if (n.prev != kNil) pool_[n.prev].next = n.next;
    else level.head = n.next;
    if (n.next != kNil) pool_[n.next].prev = n.prev;
    else level.tail = n.prev;
    level.total -= n.qty;
    free_.push_back(idx);
    if (level.head == kNil) levels.erase(levels.begin() + static_cast<std::ptrdiff_t>(li));
}

template <bool IsBuy>
[[gnu::always_inline]] inline void OrderBook::match(uint64_t id, int32_t owner, Side side, int64_t limit, int64_t& qty,
                      std::vector<Fill>& out) {
    std::vector<Level>& book = IsBuy ? asks_ : bids_;
    while (qty > 0 && !book.empty()) {
        Level& level = book.back();
        if (IsBuy ? level.price > limit : level.price < limit) break;
        while (qty > 0 && level.head != kNil) {
            Node& maker = pool_[level.head];
            const int64_t q = std::min(qty, maker.qty);
            out.push_back(Fill{id, maker.id, owner, maker.owner, side, level.price, q});
            qty -= q;
            maker.qty -= q;
            level.total -= q;
            if (maker.qty == 0) {
                index_.erase(maker.id);
                free_.push_back(level.head);
                level.head = maker.next;
                if (level.head != kNil) pool_[level.head].prev = kNil;
                else level.tail = kNil;
            }
        }
        if (level.head == kNil) book.pop_back();
    }
}

void OrderBook::submit_impl(uint64_t id, int32_t owner, Side side, bool has_price, int64_t price,
                            int64_t qty, TIF tif, std::vector<Fill>& out) {
    if (qty <= 0) throw std::invalid_argument("qty must be positive");
    if (id == IdMap::kEmpty) throw std::invalid_argument("order id 2^64-1 is reserved");
    if (index_.find(id) != kNil) throw std::invalid_argument("duplicate live order id");
    if (side == Side::Buy)
        match<true>(id, owner, side, has_price ? price : std::numeric_limits<int64_t>::max(), qty, out);
    else
        match<false>(id, owner, side, has_price ? price : std::numeric_limits<int64_t>::min(), qty, out);
    if (qty > 0 && tif == TIF::GTC && has_price) rest(id, owner, side, price, qty);
}

std::vector<Fill> OrderBook::submit_limit(uint64_t id, int32_t owner, Side side, int64_t price,
                                          int64_t qty, TIF tif) {
    std::vector<Fill> out;
    submit_impl(id, owner, side, true, price, qty, tif, out);
    return out;
}

std::vector<Fill> OrderBook::submit_market(uint64_t id, int32_t owner, Side side, int64_t qty) {
    std::vector<Fill> out;
    submit_impl(id, owner, side, false, 0, qty, TIF::IOC, out);
    return out;
}

bool OrderBook::cancel(uint64_t id, int32_t owner) {
    const int32_t idx = index_.find(id);
    if (idx == kNil || pool_[idx].owner != owner) return false;
    const bool is_bid = pool_[idx].side == Side::Buy;
    std::vector<Level>& levels = is_bid ? bids_ : asks_;
    const std::size_t li = find_level(levels, pool_[idx].price, is_bid);
    index_.erase(id);
    unlink(idx, levels, li);
    return true;
}

std::size_t OrderBook::cancel_owner(int32_t owner) {
    std::vector<uint64_t> ids;
    index_.for_each([&](uint64_t oid, int32_t idx) {
        if (pool_[idx].owner == owner) ids.push_back(oid);
    });
    for (auto oid : ids) cancel(oid, owner);
    return ids.size();
}

std::optional<int64_t> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.back().price;
}

std::optional<int64_t> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.back().price;
}

int64_t OrderBook::volume_at(Side side, int64_t price) const {
    const bool is_bid = side == Side::Buy;
    const std::vector<Level>& levels = is_bid ? bids_ : asks_;
    const std::size_t li = find_level(levels, price, is_bid);
    return li == levels.size() ? 0 : levels[li].total;
}

std::vector<std::pair<int64_t, int64_t>> OrderBook::depth(Side side, std::size_t levels) const {
    const std::vector<Level>& book = side == Side::Buy ? bids_ : asks_;
    std::vector<std::pair<int64_t, int64_t>> out;
    for (auto it = book.rbegin(); it != book.rend() && out.size() < levels; ++it)
        out.emplace_back(it->price, it->total);
    return out;
}

void OrderBook::clear() {
    pool_.clear();
    free_.clear();
    bids_.clear();
    asks_.clear();
    index_.clear();
}

// ----------------------------------------------------------------- Ledger
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

// ----------------------------------------------------------------- replay
ReplayStats replay(OrderBook& book, const std::vector<int8_t>& op, const std::vector<uint64_t>& id,
                   const std::vector<int32_t>& owner, const std::vector<int8_t>& side,
                   const std::vector<int64_t>& price, const std::vector<int64_t>& qty) {
    const std::size_t n = op.size();
    if (id.size() != n || owner.size() != n || side.size() != n || price.size() != n ||
        qty.size() != n)
        throw std::invalid_argument("replay: arrays must have equal length");
    ReplayStats st;
    uint64_t h = 1469598103934665603ULL;  // FNV-style mix of (maker_id, price, qty) per fill
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };
    std::vector<Fill> fills;
    fills.reserve(64);
    for (std::size_t i = 0; i < n; ++i) {
        const Side s = side[i] == 0 ? Side::Buy : Side::Sell;
        fills.clear();
        switch (op[i]) {
            case 0: book.submit_impl(id[i], owner[i], s, true, price[i], qty[i], TIF::GTC, fills); break;
            case 1: book.submit_impl(id[i], owner[i], s, true, price[i], qty[i], TIF::IOC, fills); break;
            case 2: book.submit_impl(id[i], owner[i], s, false, 0, qty[i], TIF::IOC, fills); break;
            case 3:
                st.cancels_ok += book.cancel(id[i], owner[i]) ? 1 : 0;
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
