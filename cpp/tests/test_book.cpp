// Minimal dependency-free C++ unit tests for the order book.
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "mmb/order_book.hpp"

using namespace mmb;

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static void test_price_time_priority() {
    OrderBook b;
    b.submit_limit(1, 10, Side::Sell, 101, 5, TIF::GTC);
    b.submit_limit(2, 11, Side::Sell, 100, 3, TIF::GTC);
    b.submit_limit(3, 12, Side::Sell, 100, 4, TIF::GTC);
    CHECK(b.best_ask() == 100);
    auto fills = b.submit_limit(4, 20, Side::Buy, 101, 10, TIF::GTC);
    CHECK(fills.size() == 3);
    CHECK(fills[0].maker_id == 2 && fills[0].qty == 3 && fills[0].price == 100);
    CHECK(fills[1].maker_id == 3 && fills[1].qty == 4 && fills[1].price == 100);
    CHECK(fills[2].maker_id == 1 && fills[2].qty == 3 && fills[2].price == 101);
    CHECK(b.volume_at(Side::Sell, 101) == 2);
    CHECK(!b.best_bid().has_value());
}

static void test_rest_and_ioc() {
    OrderBook b;
    auto f = b.submit_limit(1, 1, Side::Buy, 99, 5, TIF::IOC);
    CHECK(f.empty() && b.order_count() == 0);
    b.submit_limit(2, 1, Side::Buy, 99, 5, TIF::GTC);
    CHECK(b.best_bid() == 99 && b.order_count() == 1);
    f = b.submit_market(3, 2, Side::Sell, 8);
    CHECK(f.size() == 1 && f[0].qty == 5 && f[0].price == 99);
    CHECK(b.order_count() == 0);
}

static void test_cancel_and_errors() {
    OrderBook b;
    b.submit_limit(1, 1, Side::Buy, 99, 5, TIF::GTC);
    b.submit_limit(2, 1, Side::Buy, 99, 5, TIF::GTC);
    CHECK(b.cancel(1));
    CHECK(!b.cancel(1));
    CHECK(b.volume_at(Side::Buy, 99) == 5);
    bool threw = false;
    try { b.submit_limit(2, 1, Side::Buy, 98, 1, TIF::GTC); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { b.submit_limit(9, 1, Side::Buy, 98, 0, TIF::GTC); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    CHECK(b.cancel_owner(1) == 1 && b.order_count() == 0);
}

static void test_ledger() {
    Exchange ex;
    ex.submit_limit(1, 7, Side::Sell, 100, 4, TIF::GTC);
    ex.submit_limit(2, 8, Side::Buy, 105, 3, TIF::IOC);
    CHECK(ex.ledger.position(7) == -3 && ex.ledger.cash(7) == 300);
    CHECK(ex.ledger.position(8) == 3 && ex.ledger.cash(8) == -300);
    CHECK(ex.ledger.mark(7, 110) == 300 - 330);
}

int main() {
    test_price_time_priority();
    test_rest_and_ioc();
    test_cancel_and_errors();
    test_ledger();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all C++ tests passed\n");
    return 0;
}
