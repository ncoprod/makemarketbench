// pybind11 bindings: exposes the C++ engine as makemarket._engine.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mmb/order_book.hpp"

namespace py = pybind11;
using namespace mmb;

namespace {
// Fills are returned to Python as plain tuples (taker_id, maker_id, taker_owner,
// maker_owner, taker_side, price, qty) so both backends share one representation.
py::list fills_to_list(const std::vector<Fill>& fills) {
    py::list out;
    for (const auto& f : fills)
        out.append(py::make_tuple(f.taker_id, f.maker_id, f.taker_owner, f.maker_owner,
                                  static_cast<int>(f.taker_side), f.price, f.qty));
    return out;
}
Side to_side(int s) {
    if (s != 0 && s != 1) throw std::invalid_argument("side must be 0 (buy) or 1 (sell)");
    return s == 0 ? Side::Buy : Side::Sell;
}
}  // namespace

PYBIND11_MODULE(_engine, m) {
    m.doc() = "MakeMarketBench C++ matching engine";

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def(
            "submit_limit",
            [](OrderBook& b, uint64_t id, int32_t owner, int side, int64_t price, int64_t qty,
               bool ioc) {
                return fills_to_list(
                    b.submit_limit(id, owner, to_side(side), price, qty, ioc ? TIF::IOC : TIF::GTC));
            },
            py::arg("id"), py::arg("owner"), py::arg("side"), py::arg("price"), py::arg("qty"),
            py::arg("ioc") = false)
        .def(
            "submit_market",
            [](OrderBook& b, uint64_t id, int32_t owner, int side, int64_t qty) {
                return fills_to_list(b.submit_market(id, owner, to_side(side), qty));
            },
            py::arg("id"), py::arg("owner"), py::arg("side"), py::arg("qty"))
        .def("cancel", &OrderBook::cancel)
        .def("cancel_owner", &OrderBook::cancel_owner)
        .def("best_bid", &OrderBook::best_bid)
        .def("best_ask", &OrderBook::best_ask)
        .def("volume_at",
             [](const OrderBook& b, int side, int64_t price) { return b.volume_at(to_side(side), price); })
        .def("depth",
             [](const OrderBook& b, int side, std::size_t levels) { return b.depth(to_side(side), levels); })
        .def("order_count", &OrderBook::order_count)
        .def("has_order", &OrderBook::has_order)
        .def("clear", &OrderBook::clear)
        .def(
            "replay",
            [](OrderBook& b, const std::vector<int8_t>& op, const std::vector<uint64_t>& id,
               const std::vector<int32_t>& owner, const std::vector<int8_t>& side,
               const std::vector<int64_t>& price, const std::vector<int64_t>& qty) {
                ReplayStats st;
                {
                    py::gil_scoped_release release;
                    st = replay(b, op, id, owner, side, price, qty);
                }
                py::dict d;
                d["fills"] = st.fills;
                d["volume"] = st.volume;
                d["cancels_ok"] = st.cancels_ok;
                d["checksum"] = st.checksum;
                return d;
            },
            "Replay an operation stream natively (op: 0=GTC, 1=IOC, 2=market, 3=cancel).");

    py::class_<Ledger>(m, "Ledger")
        .def(py::init<>())
        .def(
            "apply",
            [](Ledger& l, py::tuple t) {
                Fill f{t[0].cast<uint64_t>(), t[1].cast<uint64_t>(), t[2].cast<int32_t>(),
                       t[3].cast<int32_t>(), to_side(t[4].cast<int>()), t[5].cast<int64_t>(),
                       t[6].cast<int64_t>()};
                l.apply(f);
            })
        .def("position", &Ledger::position)
        .def("cash", &Ledger::cash)
        .def("mark", &Ledger::mark)
        .def("clear", &Ledger::clear);
}
