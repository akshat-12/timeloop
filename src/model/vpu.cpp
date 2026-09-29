#include "model/vpu.hpp"

#include <limits>
#include <stdexcept>

namespace model
{
namespace
{
const std::map<std::string, VPU::VPUOp> operation_names
    = { { "add", VPU::VPUOp::ADD },        { "sub", VPU::VPUOp::SUB },
        { "mul", VPU::VPUOp::MUL },        { "relu", VPU::VPUOp::RELU },
        { "tanh", VPU::VPUOp::TANH },      { "sigmoid", VPU::VPUOp::SIGMOID },
        { "softmax", VPU::VPUOp::SOFTMAX } };

std::uint64_t
Positive(config::CompoundConfigNode node, const char* key)
{
  long long value = 0;
  if (!node.lookupValue(key, value) || value <= 0)
    throw std::invalid_argument(
        std::string("VPU requires a positive integer: ") + key);
  return static_cast<std::uint64_t>(value);
}

void
Validate(const VPU::Specs& specs)
{
  if (!specs.instances.IsSpecified() || specs.instances.Get() != 1)
    throw std::invalid_argument("VPU currently supports one shared instance");
  if (!specs.vector_width.IsSpecified() || specs.vector_width.Get() == 0)
    throw std::invalid_argument("VPU vector_width must be positive");
  if (specs.operations.empty())
    throw std::invalid_argument("VPU requires at least one operation");
  for (const auto& entry : specs.operations)
    {
      const auto& timing = entry.second;
      if (!timing.latency.IsSpecified() || timing.latency.Get() == 0
          || !timing.initiation_interval.IsSpecified()
          || timing.initiation_interval.Get() == 0)
        throw std::invalid_argument(
            "VPU operation latency and initiation_interval must be positive");
    }
}
} // namespace

VPU::VPUOp VPU::ParseOperation(const std::string& name) {
  auto it = operation_names.find(name);
  if (it == operation_names.end()) throw std::invalid_argument("Unknown VPU operation: " + name);
  return it->second;
}

VPU::Specs
VPU::ParseSpecs(config::CompoundConfigNode setting)
{
  Specs specs;
  std::string name = "VPU";
  setting.lookupValue("name", name);
  specs.name = config::parseName(name);

  auto attributes
      = setting.exists("attributes") ? setting.lookup("attributes") : setting;

  if (attributes.exists("instances"))
    specs.instances = Positive(attributes, "instances");

  attributes.lookupValue("connected_buffer", specs.connected_buffer);
  attributes.lookupValue("backing_storage", specs.backing_storage);
  specs.vector_width = Positive(attributes, "vector_width");
  if (!attributes.exists("operations"))
    throw std::invalid_argument("VPU requires an operations mapping");

  auto operations = attributes.lookup("operations");

  if (!operations.isMap())
    throw std::invalid_argument("VPU operations must be a mapping");

  std::vector<std::string> names;
  operations.getMapKeys(names);

  for (const auto& op : names)
    {
      auto found = operation_names.find(op);
      if (found == operation_names.end())
        throw std::invalid_argument("Unknown VPU operation: " + op);
      auto node = operations.lookup(op);
      OperationSpec timing;
      timing.latency = Positive(node, "latency");
      // Default to serial batches. A smaller explicit interval models
      // pipelining.
      timing.initiation_interval = node.exists("initiation_interval")
                                       ? Positive(node, "initiation_interval")
                                       : timing.latency.Get();
      specs.operations.emplace(found->second, timing);
    }
  Validate(specs);
  return specs;
}

VPU::VPU(const Specs& specs) : specs_(specs)
{
  Validate(specs_);
  is_specced_ = true;
  is_evaluated_ = false;
}

EvalStatus
VPU::Evaluate(VPUOp operation, std::uint64_t count)
{
  stats_ = Stats{};
  is_evaluated_ = false;
  auto found = specs_.operations.find(operation);
  if (found == specs_.operations.end())
    return { false, "Operation is not configured on this VPU" };

  const auto width = specs_.vector_width.Get();
  const auto batches
      = count / width + (count % width != 0); // ceil(count / width)
  const auto latency = found->second.latency.Get();
  const auto interval = found->second.initiation_interval.Get();
  std::uint64_t cycles = 0;
  if (batches != 0)
    {
      // First batch finishes after latency; each following batch finishes II
      // later.
      if (batches - 1
          > (std::numeric_limits<std::uint64_t>::max() - latency) / interval)
        return { false, "VPU cycle count overflow" };
      cycles = latency + (batches - 1) * interval;
    }
  stats_.scalar_operations = count;
  stats_.vector_operations = batches;
  stats_.cycles = cycles;
  stats_.utilization
      = batches == 0 ? 0.0
                     : static_cast<double>(static_cast<long double>(count)
                                           / batches / width);
  is_evaluated_ = true;
  return { true, "" };
}

std::shared_ptr<Level>
VPU::Clone() const
{
  return std::make_shared<VPU>(*this);
}
std::string
VPU::Name() const
{
  return specs_.name.Get();
}
std::uint64_t
VPU::Cycles() const
{
  return stats_.cycles.Get();
}
std::uint64_t
VPU::UtilizedInstances(problem::Shape::DataSpaceID) const
{
  return stats_.scalar_operations.Get() == 0 ? 0 : 1;
}
void
VPU::Print(std::ostream& out) const
{
  out << "VPU: " << Name() << '\n'
      << "  Vector width: " << specs_.vector_width.Get() << '\n'
      << "  Scalar operations: " << stats_.scalar_operations.Get() << '\n'
      << "  Vector batches: " << stats_.vector_operations.Get() << '\n'
      << "  Cycles: " << Cycles() << '\n'
      << "  Lane utilization: " << stats_.utilization.Get() << '\n'
      << "  Energy, area, and memory transfers are not modeled.\n";
}

// A generic MAC tile does not identify the VPU operation. Fail explicitly
// until topology integration calls the operation-specific Evaluate overload
// above.
EvalStatus
VPU::PreEvaluationCheck(const problem::PerDataSpace<std::size_t>,
                        const tiling::CompoundMask, const problem::Workload*,
                        const sparse::PerStorageLevelCompressionInfo,
                        const double, const bool)
{
  return { false, "VPU requires explicit operation/count evaluation" };
}
EvalStatus
VPU::Evaluate(const tiling::CompoundTile&, const tiling::CompoundMask&,
              problem::Workload*, const double, const std::uint64_t,
              const bool)
{
  stats_ = Stats{};
  is_evaluated_ = false;
  return { false, "Call VPU::Evaluate(operation, count); MAC-tile integration "
                  "is not implemented" };
}
} // namespace model
