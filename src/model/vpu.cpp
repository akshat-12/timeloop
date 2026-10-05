// A configurable operation cost, not numerical execution or a TPU model.
#include "model/vpu.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

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
  if (!s.input_width || !s.buffer_slots || s.input_width != pe_width)
    throw std::invalid_argument(
        "PE output_width must equal positive VPU input_width; buffer_slots must be positive");

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

VPU::Stats VPU::Schedule(const Specs& specs, const std::vector<std::array<U, 8>>& tile_costs,
                         bool has_shared_buffer_port)
{
  if (!specs.buffer_slots)
    throw std::invalid_argument("VPU buffer_slots must be positive");

  Stats stats;
  stats.active = true;
  stats.tiles = tile_costs.size();

  // Each slot records when its previous tile has finished writing to DRAM.
  // Reserving complete tile lifetimes prevents unlimited prefetching.
  std::vector<U> slot_available_at(specs.buffer_slots, 0);

  // Resource calendars store non-overlapping [start, finish) intervals.
  std::map<std::string, std::map<U, U>> resource_reservations;

  // These track when each unit may accept its next tile, including handoff
  // and output-write stalls that extend beyond its compute phase.
  U pe_available_at = 0, vpu_available_at = 0;

  // Find the earliest interval that is free on every required resource.
  // Entire tiles are scheduled one at a time, so an earlier tile may already
  // reserve a future interval. Later tiles can still fill gaps before it.
  auto schedule_event =
      [&](U tile, const std::string& phase, U ready_at, U duration, std::vector<std::string> resources)
  {
    U start = ready_at;

    // Zero-duration phases preserve dependency timing but reserve no resource.
    if (duration)
    {
      for (;;)
      {
        U next_start = start;
        for (const auto& resource : resources)
        {
          auto& reservations = resource_reservations[resource];
          auto reservation = reservations.lower_bound(start);

          // The preceding reservation may extend across our proposed start.
          if (reservation != reservations.begin())
            next_start = std::max(next_start, std::prev(reservation)->second);

          // Later reservations can also conflict with the proposed duration.
          for (; reservation != reservations.end() && reservation->first < Add(start, duration); ++reservation)
            next_start = std::max(next_start, reservation->second);
        }

        if (next_start == start)
          break;

        // Moving past a conflict can introduce a conflict on another resource;
        // check all calendars again until the whole interval is available.
        start = next_start;
      }

      // Commit only after finding a common gap. Intervals are half-open, so
      // a resource can be reused at the exact cycle a previous event finishes.
      for (const auto& resource : resources)
        resource_reservations[resource].emplace(start, Add(start, duration));
    }

    U finish = Add(start, duration);

    // Elapsed time includes overlap; serial time sums the same phase costs
    // as if no two phases could overlap.
    stats.total = std::max(stats.total, finish);
    stats.serial = Add(stats.serial, duration);

    // Bound report size without truncating the schedule or its cycle totals.
    if (duration && stats.trace.size() < 256)
      stats.trace.push_back({tile, start, finish, phase, resources});

    return finish;
  };

  // Memory transfers share ports; DRAM reads and writes also share one DMA.
  // A direct VPU-to-DRAM write uses DMA but does not occupy a buffer port.
  auto schedule_transfer =
      [&](U tile, const std::string& phase, U ready_at, U duration, const std::string& buffer_port, bool uses_dma)
  {
    std::vector<std::string> resources;
    if (!buffer_port.empty())
    {
      resources.push_back(buffer_port);
      if (has_shared_buffer_port)
        resources.push_back("buffer_shared");
    }

    if (uses_dma)
      resources.push_back("dma");

    return schedule_event(tile, phase, ready_at, duration, resources);
  };

  for (U tile = 0; tile < tile_costs.size(); ++tile)
  {
    // Phase order matches the cost array assembled by Topology::EvaluateVPU:
    // prefetch, PE read, PE compute, PE write, VPU load, VPU compute,
    // VPU output write, and optional buffer-to-DRAM drain.
    const auto& phase_cycles = tile_costs[tile];

    // Phases 0-3: wait for this slot, fetch operands, then run the producer.
    // Prefetch may overlap previous tiles; PE reads wait for producer release.
    auto prefetch_finish = schedule_transfer(tile, "prefetch", slot_available_at[tile % specs.buffer_slots], phase_cycles[0], "buffer_write", true);
    auto pe_read_finish = schedule_transfer(tile, "pe_read", std::max(prefetch_finish, pe_available_at), phase_cycles[1], "buffer_read", false);
    auto pe_compute_finish = schedule_event(tile, "pe", pe_read_finish, phase_cycles[2], {"pe"});
    auto pe_write_finish = schedule_transfer(tile, "pe_write", pe_compute_finish, phase_cycles[3], "buffer_write", false);

    // Phase 4: input routing determines when the PE is released for its next tile.
    // Both routes wait until the VPU has finished writing its previous result.
    U vpu_load_finish;
    if (specs.input_source == "pe")
    {
      // One completed output tile can stay at the PE; no unbounded hidden FIFO.
      vpu_load_finish = schedule_event(tile, "pe_to_vpu", std::max(pe_write_finish, vpu_available_at), phase_cycles[4], {"pe_output", "vpu_input"});

      // Block the next PE tile until the current result has been accepted.
      pe_available_at = vpu_load_finish;
    }
    else
    {
      vpu_load_finish = schedule_transfer(tile, "vpu_read", std::max(pe_write_finish, vpu_available_at), phase_cycles[4], "buffer_read", false);

      // Buffered results release the PE as soon as its buffer write completes.
      pe_available_at = pe_write_finish;
    }

    // Phases 5-6: VPU compute is followed by output transfer. Keep the VPU
    // unavailable until that write completes, even if its compute has finished.
    auto vpu_compute_finish = schedule_event(tile, "vpu", vpu_load_finish, phase_cycles[5], {"vpu"});
    bool direct_dram_output = specs.output_destination == "dram";
    vpu_available_at = schedule_transfer(tile, direct_dram_output ? "vpu_to_dram" : "vpu_write", vpu_compute_finish, phase_cycles[6],
                        direct_dram_output ? "" : "buffer_write", direct_dram_output);

    // Phase 7: buffered output needs a final drain; direct output is already
    // in DRAM. Only then may a later tile reuse this slot and its working set.
    slot_available_at[tile % specs.buffer_slots] =
        direct_dram_output ? vpu_available_at : schedule_transfer(tile, "drain", vpu_available_at, phase_cycles[7], "buffer_read", true);
    stats.compute = Add(stats.compute, phase_cycles[5]);
    stats.write = Add(stats.write, phase_cycles[6]);
  }

  return stats;
}
} // namespace model
