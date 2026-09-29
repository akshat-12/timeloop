// Akshat: VPU interface: configurable operation timing and per-piece statistics.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "compound-config/compound-config.hpp"
#include "model/attribute.hpp"
#include "model/level.hpp"

namespace model {

// Akshat: Compute-only model; Topology::EvaluateVPU handles the connected memory flow.
class VPU : public Level {
 public:
    // Akshat: Operation names supported by the standalone compute model.
    enum class VPUOp { ADD, SUB, MUL, RELU, TANH, SIGMOID, SOFTMAX };

    // Akshat: Latency is completion delay; initiation interval is batch-start spacing.
    struct OperationSpec {
    Attribute<std::uint64_t> latency;
    Attribute<std::uint64_t> initiation_interval;
    };

    // Akshat: Hardware attributes, including names resolved by topology for memory connections.
    struct Specs {
    Attribute<std::string> name = std::string("VPU");
    Attribute<std::uint64_t> instances = std::uint64_t(1);
    Attribute<std::uint64_t> vector_width;
    std::string connected_buffer = "shared_buffer", backing_storage = "DRAM";
    // Each supported operation has its own timing parameters.
    std::map<VPUOp, OperationSpec> operations;
    };

    // Akshat: Results for the most recently evaluated piece; topology sums piece costs.
    struct Stats {
    Attribute<std::uint64_t> scalar_operations = std::uint64_t(0);
    Attribute<std::uint64_t> vector_operations = std::uint64_t(0);
    Attribute<std::uint64_t> cycles = std::uint64_t(0);
    Attribute<double> utilization = 0.0; // Occupied lanes / available lanes.
    };

    // Akshat: Public construction, YAML parsing, and operation-name lookup.
    explicit VPU(const Specs& specs);
    static Specs ParseSpecs(config::CompoundConfigNode setting);
    static VPUOp ParseOperation(const std::string& name);

    // Akshat: Entry point used by the connected topology after a complete output tile is ready.
    // Evaluate one piece, replacing the previous statistics (not accumulating).
    // count means independent invocations: elements for elementwise operations,
    // complete vectors for softmax. Softmax latency must include its whole vector.
    EvalStatus Evaluate(VPUOp operation, std::uint64_t count);
    const Stats& GetStats() const { return stats_; }

    // Akshat: Level compatibility methods; zero energy/area values are placeholders, not estimates.
    // Required Level interface. No memory, energy, or area model at this stage.
    std::shared_ptr<Level> Clone() const override;
    std::string Name() const override;
    std::uint64_t Cycles() const override;
    void Print(std::ostream& out) const override;
    bool HardwareReductionSupported() override { return false; }
    double Area() const override { return 0; }
    double AreaPerInstance() const override { return 0; }
    double CapacityUtilization() const override { return 0; }
    double Energy(problem::Shape::DataSpaceID = 0) const override { return 0; }
    std::uint64_t Accesses(problem::Shape::DataSpaceID = 0) const override { return 0; }
    std::uint64_t UtilizedCapacity(problem::Shape::DataSpaceID = 0) const override { return 0; }
    std::uint64_t TileSize(problem::Shape::DataSpaceID = 0) const override { return 0; }
    std::uint64_t UtilizedInstances(problem::Shape::DataSpaceID = 0) const override;

    // Akshat: Generic MAC-tile entry points are deliberately unsupported; use the explicit operation/count overload.
    EvalStatus PreEvaluationCheck(const problem::PerDataSpace<std::size_t>,
                                const tiling::CompoundMask, const problem::Workload*,
                                const sparse::PerStorageLevelCompressionInfo,
                                const double, const bool) override;
    EvalStatus Evaluate(const tiling::CompoundTile&, const tiling::CompoundMask&,
                        problem::Workload*, const double, const std::uint64_t,
                        const bool) override;

 private:
    Specs specs_;
    Stats stats_;
};
} // namespace model
