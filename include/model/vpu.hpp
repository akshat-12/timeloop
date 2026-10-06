// Aggregate VPU throughput model. All widths are elements.
#pragma once
#include <cstdint>
#include <map>
#include <string>

namespace model
{
class VPU
{
public:
  using U = std::uint64_t;

  struct Specs
  {
    bool enabled = false;
    std::string name;

    std::string buffer = "shared_buffer";
    std::string dram = "DRAM";

    std::string input_source = "shared_buffer";
    std::string output_destination = "shared_buffer";

    U input_width = 0;  // Elements accepted per batch; must match the PE output width.
    std::map<std::string, U> latencies; // Cycles for one batch of each operation.
  };

  struct Stats
  {
    bool active = false;

    U tiles = 0;
    U elements = 0; // Output elements in one completed PE tile.
    U batches = 0;  // VPU batches per tile.

    U buffer_to_vpu = 0;
    U pe_to_vpu = 0;
    U compute = 0;
    U vpu_to_buffer = 0;
    U vpu_to_dram = 0;
    U buffer_cycles = 0;
    U dram_cycles = 0;
    U total = 0; // Maximum resource/route cost; assumes full overlap.

    // Route-specific traffic counters count logical words, not bytes.
    U pe_final_write_words = 0;
    U vpu_read_words = 0;
    U vpu_write_words = 0;
    U drain_read_words = 0;
    U dram_write_words = 0;
  };

  // Checked arithmetic for traffic and batch costs.
  static U Add(U a, U b);
  static U Mul(U a, U b);
  static U Batches(U elements, U width);

  static void Validate(const Specs& specs, U pe_output_width);
  static U Cycles(const Specs& specs, const std::string& operation, U elements);
};
} // namespace model
