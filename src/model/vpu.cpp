// A configurable operation cost, not numerical execution or a TPU model.
#include "model/vpu.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace model
{

// This adds timing/count values; it does not model the VPU's ADD operation.
// Check before adding so unsigned wraparound cannot produce a false short runtime.
VPU::U VPU::Add(U a, U b)
{
  if (b > std::numeric_limits<U>::max() - a)
    throw std::overflow_error("VPU cycle overflow");

  return a + b;
}

// Multiplication is used for batch counts and cycle totals. The b != 0 guard
// avoids division by zero; a zero operand always produces a safe zero product.
VPU::U VPU::Mul(U a, U b)
{
  if (b && a > std::numeric_limits<U>::max() / b)
    throw std::overflow_error("VPU count overflow");

  return a * b;
}

// Round up without computing n + w - 1, which could itself overflow.
// A partially filled final batch still occupies the VPU for a full batch latency.
VPU::U VPU::Batches(U n, U w)
{
  if (!w)
    throw std::invalid_argument("VPU input_width must be positive");

  return n / w + (n % w != 0);
}

void VPU::Validate(const Specs& s, U pe_width)
{
  if (!s.input_width || s.input_width != pe_width)
    throw std::invalid_argument(
        "PE output_width must equal positive VPU input_width");
  if (s.input_source != "pe" && s.input_source != "shared_buffer")
    throw std::invalid_argument("VPU input_source must be pe or shared_buffer");

  if (s.output_destination != "dram" && s.output_destination != "shared_buffer")
    throw std::invalid_argument("VPU output_destination must be dram or shared_buffer");

  const std::vector<std::string> names = {"add",     "sub",  "mul",    "relu",
                                          "sigmoid", "tanh", "softmax"};
  if (s.latencies.empty())
    throw std::invalid_argument("VPU needs operation latencies");

  for (const auto& op : s.latencies)
    if (!op.second || std::find(names.begin(), names.end(), op.first) == names.end())
      throw std::invalid_argument("Unknown VPU operation or nonpositive latency: " + op.first);
}

VPU::U VPU::Cycles(const Specs& s, const std::string& op, U elements)
{
  auto it = s.latencies.find(op);
  if (it == s.latencies.end())
    throw std::invalid_argument("Missing VPU latency: " + op);

  // Deliberately no internal pipelining: one batch finishes before the next starts.

  return Mul(Batches(elements, s.input_width), it->second);
}

} // namespace model
