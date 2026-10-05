// Simple batch VPU and bounded tile pipeline. All widths are elements.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

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
    U buffer_slots = 2; // Maximum number of tile working sets alive simultaneously.

    std::map<std::string, U> latencies; // Cycles for one batch of each operation.
  };

  struct Event
  {
    U tile;
    U start;
    U finish; // Exclusive end: a resource can be reused at this cycle.

    std::string phase;
    std::vector<std::string> resources;
  };

  struct Stats
  {
    bool active = false;

    U tiles = 0;
    U elements = 0; // Output elements in one completed PE tile.
    U batches = 0;  // VPU batches per tile.

    // Service times add all tiles' work; total measures elapsed time with overlap.
    U compute = 0;
    U write = 0;
    U total = 0;
    U serial = 0;

    // Route-specific traffic counters count logical words, not bytes.
    U reserved_words = 0;
    U pe_final_write_words = 0;
    U vpu_read_words = 0;
    U vpu_write_words = 0;
    U drain_read_words = 0;
    U dram_write_words = 0;

    std::vector<Event> trace;
  };

  // Checked arithmetic is shared by transfer, batch and schedule costs.
  static U Add(U a, U b);
  static U Mul(U a, U b);
  static U Batches(U elements, U width);

  static void Validate(const Specs& specs, U pe_output_width);
  static U Cycles(const Specs& specs, const std::string& operation, U elements);

  // Eight phases: prefetch, PE read/compute/write, VPU read/compute/write, drain.
  // Direct input holds the PE until handoff; output holds the VPU until its write finishes.
  static Stats Schedule(const Specs& specs, const std::vector<std::array<U, 8>>& costs,
                        bool shared_port);
};
} // namespace model
