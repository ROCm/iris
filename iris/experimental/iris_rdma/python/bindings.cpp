// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

/******************************************************************************
 * Python Bindings for Iris RDMA Backend using PyBind11
 *****************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "network_backend.hpp"
#include "queue_pair.hpp"
#include "torch_bootstrap.hpp"
#include "iris_manager.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_iris_rdma_backend, m) {
  m.doc() =
      "Iris RDMA Backend: InfiniBand RDMA with PyTorch Integration";

  // Expose NICVendor enum
  py::enum_<iris::rdma::nic_vendor>(m, "nic_vendor")
      .value("NONE", iris::rdma::nic_vendor::NONE)
      .value("IONIC", iris::rdma::nic_vendor::IONIC)
      .value("BNXT", iris::rdma::nic_vendor::BNXT)
      .value("MLX5", iris::rdma::nic_vendor::MLX5)
      .export_values();

  // Expose qp_info_t struct
  py::class_<iris::rdma::qp_info_t>(m, "qp_info_t")
      .def(py::init<>())
      .def_readwrite("qp_num", &iris::rdma::qp_info_t::qp_num)
      .def_readwrite("lkey", &iris::rdma::qp_info_t::lkey)
      .def_readwrite("rkey", &iris::rdma::qp_info_t::rkey)
      .def_readwrite("dst_rank", &iris::rdma::qp_info_t::dst_rank)
      .def("__repr__", [](const iris::rdma::qp_info_t& info) {
        return "<qp_info_t qp_num=" + std::to_string(info.qp_num) +
               " lkey=" + std::to_string(info.lkey) +
               " rkey=" + std::to_string(info.rkey) +
               " dst_rank=" + std::to_string(info.dst_rank) + ">";
      });

  // Expose torch_bootstrap
  py::class_<iris::rdma::torch_bootstrap, std::shared_ptr<iris::rdma::torch_bootstrap>>(m,
                                                              "torch_bootstrap")
      .def(py::init([](py::object pg_obj) {
             // Extract c10d::ProcessGroup from Python object
             auto pg_ptr =
                 pg_obj.cast<c10::intrusive_ptr<c10d::ProcessGroup>>();
             return std::make_shared<iris::rdma::torch_bootstrap>(pg_ptr);
           }),
           py::arg("process_group"))
      .def("get_rank", &iris::rdma::torch_bootstrap::get_rank)
      .def("get_world_size", &iris::rdma::torch_bootstrap::get_world_size)
      .def("barrier", &iris::rdma::torch_bootstrap::barrier);

  // Expose queue_pair (read-only access)
  py::class_<iris::queue_pair>(m, "queue_pair")
      .def("get_qp_num", &iris::queue_pair::get_qp_num)
      .def("get_lkey", &iris::queue_pair::get_lkey)
      .def("get_rkey", &iris::queue_pair::get_rkey)
      .def("get_dst_rank", &iris::queue_pair::get_dst_rank)
      .def("get_info", &iris::queue_pair::get_info)
      .def("__repr__", [](const iris::queue_pair& qp) {
        return "<queue_pair qp_num=" + std::to_string(qp.get_qp_num()) +
               " dst_rank=" + std::to_string(qp.get_dst_rank()) + ">";
      });

  // Expose network_backend
  py::class_<iris::network_backend>(m, "network_backend")
      .def(py::init<std::shared_ptr<iris::rdma::torch_bootstrap>, const char*>(),
           py::arg("bootstrap"), py::arg("device_name") = nullptr,
           "Create network_backend with PyTorch bootstrap")
      .def("init", &iris::network_backend::init,
           "Initialize the network (setup QPs, transition to RTS)")
      .def(
          "register_memory",
          [](iris::network_backend& self, py::object obj, size_t size = 0) {
            void* ptr = nullptr;
            size_t actual_size = size;
            
            // Check if it's an integer (raw pointer)
            if (PyLong_Check(obj.ptr())) {
              ptr = reinterpret_cast<void*>(PyLong_AsVoidPtr(obj.ptr()));
              if (size == 0) {
                throw std::runtime_error("Size must be specified for raw pointer");
              }
              actual_size = size;
            }
            // Check if it's a PyTorch tensor
            else if (THPVariable_Check(obj.ptr())) {
              auto t = THPVariable_Unpack(obj.ptr());
              ptr = t.data_ptr();
              actual_size = t.numel() * t.element_size();
            }
            else {
              throw std::runtime_error("Expected a PyTorch tensor or integer address");
            }
            
            self.register_memory(ptr, actual_size);
          },
          py::arg("obj"), py::arg("size") = 0,
          "Register memory for RDMA (supports CPU pinned or GPU memory via GPUDirect)")
      .def("get_qp", &iris::network_backend::get_qp, py::arg("dst_rank"),
           py::return_value_policy::reference_internal,
           "Get queue pair for destination rank")
      .def("get_qp_info", &iris::network_backend::get_qp_info, py::arg("dst_rank"),
           "Get QP info for destination rank")
      .def("get_rank", &iris::network_backend::get_rank, "Get rank")
      .def("get_world_size", &iris::network_backend::get_world_size, "Get world size")
      .def("get_remote_heap_base", &iris::network_backend::get_remote_heap_base,
           py::arg("rank"),
           "Get remote heap base address for a rank")
      .def("get_heap_base", &iris::network_backend::get_heap_base,
           "Get local heap base address")
      .def("get_heap_size", &iris::network_backend::get_heap_size,
           "Get heap size in bytes")
      .def("rdma_write",
           [](iris::network_backend& self, int dst_rank, uint64_t local_addr,
              uint64_t remote_addr, size_t size, uint64_t wr_id) {
             return self.rdma_write(dst_rank, reinterpret_cast<void*>(local_addr),
                                   remote_addr, size, wr_id);
           },
           py::arg("dst_rank"), py::arg("local_addr"), py::arg("remote_addr"),
           py::arg("size"), py::arg("wr_id") = 0,
           "RDMA write to remote rank (local_addr is integer address)")
      .def("rdma_read",
           [](iris::network_backend& self, int dst_rank, uint64_t local_addr,
              uint64_t remote_addr, size_t size, uint64_t wr_id) {
             return self.rdma_read(dst_rank, reinterpret_cast<void*>(local_addr),
                                  remote_addr, size, wr_id);
           },
           py::arg("dst_rank"), py::arg("local_addr"), py::arg("remote_addr"),
           py::arg("size"), py::arg("wr_id") = 0,
           "RDMA read from remote rank (local_addr is integer address)")
      .def("poll_cq", &iris::network_backend::poll_cq,
           py::arg("dst_rank"), py::arg("max_completions") = 1,
           "Poll completion queue for RDMA operations")
      .def("__repr__", [](const iris::network_backend& backend) {
        return "<network_backend rank=" + std::to_string(backend.get_rank()) +
               " world_size=" + std::to_string(backend.get_world_size()) + ">";
      });

  py::class_<iris::rdma_proxy>(m, "rdma_proxy")
      .def(py::init([](std::shared_ptr<iris::rdma::torch_bootstrap> bootstrap, py::object heap_tensor, int queue_size) {
        // Extract heap pointer from tensor
        if (!THPVariable_Check(heap_tensor.ptr())) {
          throw std::runtime_error("heap_tensor must be a PyTorch tensor");
        }
        auto heap = THPVariable_Unpack(heap_tensor.ptr());
        void* heap_ptr = heap.data_ptr();
        size_t heap_size = heap.numel() * heap.element_size();
        
        return new iris::rdma_proxy(bootstrap, heap_ptr, heap_size, queue_size);
      }),
      py::arg("bootstrap"), py::arg("heap_tensor"), py::arg("queue_size") = 512,
      "Create rdma_proxy with network_backend + Queue + Proxy Thread")
      .def("start_proxy_thread", &iris::rdma_proxy::start_proxy_thread,
           "Start proxy thread that processes RDMA operations from queue")
      .def("stop_proxy_thread", &iris::rdma_proxy::stop_proxy_thread,
           "Stop proxy thread")
      .def("get_queue_ptr", 
           [](iris::rdma_proxy& self) {
             return reinterpret_cast<uintptr_t>(self.get_queue_ptr());
           },
           "Get queue pointer for Triton kernels")
      .def("get_heap_base", &iris::rdma_proxy::get_heap_base,
           "Get local heap base address")
      .def("get_remote_heap_base", &iris::rdma_proxy::get_remote_heap_base,
           py::arg("rank"),
           "Get remote heap base address for a rank")
      .def("get_rank", &iris::rdma_proxy::get_rank, "Get rank")
      .def("get_world_size", &iris::rdma_proxy::get_world_size, "Get world size")
      .def("is_queue_empty", &iris::rdma_proxy::is_queue_empty, 
           "Check if queue is empty (all work items processed)")
      .def("rdma_write",
           [](iris::rdma_proxy& self, int dst_rank, uint64_t local_addr,
              uint64_t remote_addr, size_t size, uint64_t wr_id) {
             auto backend = self.get_backend();
             return backend->rdma_write(dst_rank, reinterpret_cast<void*>(local_addr),
                                       remote_addr, size, wr_id);
           },
           py::arg("dst_rank"), py::arg("local_addr"), py::arg("remote_addr"),
           py::arg("size"), py::arg("wr_id") = 0,
           "RDMA write to remote rank (local_addr is integer address)")
      .def("rdma_read",
           [](iris::rdma_proxy& self, int dst_rank, uint64_t local_addr,
              uint64_t remote_addr, size_t size, uint64_t wr_id) {
             auto backend = self.get_backend();
             return backend->rdma_read(dst_rank, reinterpret_cast<void*>(local_addr),
                                      remote_addr, size, wr_id);
           },
           py::arg("dst_rank"), py::arg("local_addr"), py::arg("remote_addr"),
           py::arg("size"), py::arg("wr_id") = 0,
           "RDMA read from remote rank (local_addr is integer address)")
      .def("poll_cq",
           [](iris::rdma_proxy& self, int dst_rank, int max_completions) {
             auto backend = self.get_backend();
             return backend->poll_cq(dst_rank, max_completions);
           },
           py::arg("dst_rank"), py::arg("max_completions") = 1,
           "Poll completion queue for RDMA operations")
      .def("__repr__", [](const iris::rdma_proxy& mgr) {
        return "<rdma_proxy rank=" + std::to_string(mgr.get_rank()) +
               " world_size=" + std::to_string(mgr.get_world_size()) + ">";
      });
}

