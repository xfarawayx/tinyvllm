#include <torch/extension.h>
#include <ATen/core/ivalue.h>

#include "tvllm/engine.h"

static tvllm::Engine* create_engine_from_state(const std::string& model_dir,
                                               const pybind11::dict& state_dict) {
  c10::impl::GenericDict dict(c10::StringType::get(), c10::TensorType::get());
  for (auto item : state_dict) {
    auto key = item.first.cast<std::string>();
    auto tensor = item.second.cast<torch::Tensor>();
    dict.insert(c10::IValue(key), c10::IValue(tensor));
  }
  return new tvllm::Engine(model_dir, dict);
}

PYBIND11_MODULE(tinyvllm, m) {
  // Register FlashInfer's custom CUDA ops into the torch dispatcher so that
  // the attention wrapper in attention.cpp can invoke them.
  pybind11::module_::import("flashinfer");


  pybind11::class_<tvllm::SampleParams>(m, "SampleParams")
      .def(pybind11::init<>())
      .def(pybind11::init([](int64_t max_new_tokens,
                             double temperature,
                             bool ignore_eos) {
        tvllm::SampleParams p;
        p.max_new_tokens = max_new_tokens;
        p.temperature = temperature;
        p.ignore_eos = ignore_eos;
        return p;
      }),
      pybind11::arg("max_new_tokens"),
      pybind11::arg("temperature") = 0.0,
      pybind11::arg("ignore_eos") = false)
      .def_readwrite("max_new_tokens", &tvllm::SampleParams::max_new_tokens)
      .def_readwrite("temperature", &tvllm::SampleParams::temperature)
      .def_readwrite("ignore_eos", &tvllm::SampleParams::ignore_eos);

  pybind11::class_<tvllm::Engine>(m, "Engine")
      .def(pybind11::init<const std::string&>())
      .def(pybind11::init(&create_engine_from_state))
      .def("generate", &tvllm::Engine::generate,
           pybind11::arg("batch_input_ids"),
           pybind11::arg("sample_params"),
           pybind11::arg("max_batch_size") = 0)
      .def("reset", &tvllm::Engine::reset)
      .def("eos_token_id", &tvllm::Engine::eos_token_id)
      .def("max_position_embeddings", &tvllm::Engine::max_position_embeddings);
}
