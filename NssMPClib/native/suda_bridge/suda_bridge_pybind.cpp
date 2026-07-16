#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "include/batch_pir_to_share.h"

namespace py = pybind11;

namespace {

constexpr std::int64_t kSudaPrime = 1337006139375617LL;

std::vector<std::vector<std::int64_t>> array_to_feature_major(py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> array)
{
    py::buffer_info info = array.request();
    if (info.ndim != 2) {
        throw std::runtime_error("feature_major must be a 2-D int64 array");
    }
    const auto feature_num = static_cast<std::size_t>(info.shape[0]);
    const auto host_n_data = static_cast<std::size_t>(info.shape[1]);
    const auto *ptr = static_cast<const std::int64_t *>(info.ptr);
    std::vector<std::vector<std::int64_t>> data(feature_num, std::vector<std::int64_t>(host_n_data));
    for (std::size_t feature = 0; feature < feature_num; ++feature) {
        for (std::size_t row = 0; row < host_n_data; ++row) {
            data[feature][row] = ptr[feature * host_n_data + row] % kSudaPrime;
        }
    }
    return data;
}

std::vector<std::int64_t> array_to_ids(py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> array)
{
    py::buffer_info info = array.request();
    if (info.ndim != 1) {
        throw std::runtime_error("query_ids must be a 1-D int64 array");
    }
    const auto batch_size = static_cast<std::size_t>(info.shape[0]);
    const auto *ptr = static_cast<const std::int64_t *>(info.ptr);
    return std::vector<std::int64_t>(ptr, ptr + batch_size);
}

py::list vector_matrix_to_py(const std::vector<std::vector<std::int64_t>> &matrix)
{
    py::list outer;
    for (const auto &row : matrix) {
        py::list inner;
        for (const auto value : row) {
            inner.append(value);
        }
        outer.append(inner);
    }
    return outer;
}

std::stringstream bytes_to_stream(py::bytes bytes)
{
    std::string value = bytes;
    std::stringstream stream;
    stream << value;
    return stream;
}

py::bytes stream_to_bytes(std::stringstream &stream)
{
    return py::bytes(stream.str());
}

} // namespace

class BatchPirToShareClientBridge {
private:
    BatchPirToShareClient client;
    std::size_t batch_size;

public:
    BatchPirToShareClientBridge(std::size_t host_log_n_data, std::size_t feature_num, std::size_t batch_size)
        : client(host_log_n_data, feature_num, batch_size), batch_size(batch_size)
    {
    }

    py::bytes save_keys()
    {
        auto keys = client.save_keys();
        return stream_to_bytes(keys);
    }

    py::dict gen_query(py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> query_ids)
    {
        auto ids = array_to_ids(query_ids);
        if (ids.size() != batch_size) {
            throw std::runtime_error("query_ids size must equal batch_size");
        }
        auto query = client.gen_query_stream(ids);
        py::dict out;
        out["cipher_x_powers"] = stream_to_bytes(query.cipher_x_powers_stream);
        out["row_keepers"] = stream_to_bytes(query.row_keepers_stream);
        out["col_keepers"] = stream_to_bytes(query.col_keepers_stream);
        out["query_bytes"] = query.byte_size();
        return out;
    }

    py::list extract_answer(py::bytes response_bytes)
    {
        auto response = bytes_to_stream(response_bytes);
        return vector_matrix_to_py(client.extract_answer(response));
    }

    std::int64_t get_prime_num()
    {
        return client.get_prime_num();
    }
};

class BatchPirToShareServerBridge {
private:
    std::unique_ptr<BatchPirToShareServer> server;

public:
    BatchPirToShareServerBridge(
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> feature_major,
        std::size_t batch_size,
        bool use_out_mem)
    {
        auto data = array_to_feature_major(feature_major);
        server = std::make_unique<BatchPirToShareServer>(data, batch_size, data.size(), use_out_mem);
    }

    void load_keys(py::bytes keys_bytes)
    {
        auto keys = bytes_to_stream(keys_bytes);
        server->load_keys(keys);
    }

    py::dict gen_response(py::dict query_dict, bool mod_switch)
    {
        BatchPirToShareQueryStream query;
        query.cipher_x_powers_stream = bytes_to_stream(query_dict["cipher_x_powers"].cast<py::bytes>());
        query.row_keepers_stream = bytes_to_stream(query_dict["row_keepers"].cast<py::bytes>());
        query.col_keepers_stream = bytes_to_stream(query_dict["col_keepers"].cast<py::bytes>());
        std::stringstream response = server->gen_response(query, mod_switch);
        py::dict out;
        out["response"] = stream_to_bytes(response);
        out["response_bytes"] = response.str().size();
        return out;
    }

    py::list extract_answer()
    {
        return vector_matrix_to_py(server->extract_answer());
    }

    std::int64_t get_prime_num()
    {
        return server->get_prime_num();
    }
};

py::dict batch_pir_to_share(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> feature_major,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> query_ids,
    std::size_t database_size,
    std::size_t host_log_n_data,
    std::size_t batch_size,
    bool mod_switch,
    bool use_out_mem)
{
    auto data = array_to_feature_major(feature_major);
    auto ids = array_to_ids(query_ids);
    if (data.empty()) {
        throw std::runtime_error("feature_major must contain at least one feature");
    }
    if (ids.size() != batch_size) {
        throw std::runtime_error("query_ids size must equal batch_size");
    }
    const auto feature_num = data.size();

    auto t0 = std::chrono::steady_clock::now();
    BatchPirToShareServer server(data, batch_size, feature_num, use_out_mem);
    auto t1 = std::chrono::steady_clock::now();
    BatchPirToShareClient client(host_log_n_data, feature_num, batch_size);
    auto keys = client.save_keys();
    server.load_keys(keys);
    auto t2 = std::chrono::steady_clock::now();
    BatchPirToShareQueryStream query = client.gen_query_stream(ids);
    auto t3 = std::chrono::steady_clock::now();
    std::stringstream response = server.gen_response(query, mod_switch);
    auto server_share = server.extract_answer();
    auto t4 = std::chrono::steady_clock::now();
    auto client_share = client.extract_answer(response);
    auto t5 = std::chrono::steady_clock::now();

    auto ms = [](auto start, auto end) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    };

    py::dict meta;
    meta["server_init_ms"] = ms(t0, t1);
    meta["client_init_and_key_ms"] = ms(t1, t2);
    meta["query_ms"] = ms(t2, t3);
    meta["response_ms"] = ms(t3, t4);
    meta["extract_ms"] = ms(t4, t5);

    py::dict out;
    out["server_share"] = vector_matrix_to_py(server_share);
    out["client_share"] = vector_matrix_to_py(client_share);
    out["modulus"] = client.get_prime_num();
    out["query_bytes"] = query.byte_size();
    out["response_bytes"] = response.str().size();
    out["database_size"] = database_size;
    out["meta"] = meta;
    return out;
}

PYBIND11_MODULE(_suda_bridge, m)
{
    m.doc() = "Python bridge for sls33/Suda BatchPirToShareServer/Client";
    py::class_<BatchPirToShareClientBridge>(m, "BatchPirToShareClientBridge")
        .def(py::init<std::size_t, std::size_t, std::size_t>())
        .def("save_keys", &BatchPirToShareClientBridge::save_keys)
        .def("gen_query", &BatchPirToShareClientBridge::gen_query)
        .def("extract_answer", &BatchPirToShareClientBridge::extract_answer)
        .def("get_prime_num", &BatchPirToShareClientBridge::get_prime_num);
    py::class_<BatchPirToShareServerBridge>(m, "BatchPirToShareServerBridge")
        .def(py::init<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>, std::size_t, bool>())
        .def("load_keys", &BatchPirToShareServerBridge::load_keys)
        .def("gen_response", &BatchPirToShareServerBridge::gen_response)
        .def("extract_answer", &BatchPirToShareServerBridge::extract_answer)
        .def("get_prime_num", &BatchPirToShareServerBridge::get_prime_num);
    m.def(
        "batch_pir_to_share",
        &batch_pir_to_share,
        py::arg("feature_major"),
        py::arg("query_ids"),
        py::arg("database_size"),
        py::arg("host_log_n_data"),
        py::arg("batch_size"),
        py::arg("mod_switch") = true,
        py::arg("use_out_mem") = false);
}
