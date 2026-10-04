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

VPU::Stats VPU::Schedule(const Specs& s, const std::vector<std::array<U, 8>>& costs,
                         bool shared_port)
{
  if (!s.buffer_slots)
    throw std::invalid_argument("VPU buffer_slots must be positive");
  Stats out;
  out.active = true;
  out.tiles = costs.size();
  // Each slot records when its previous tile has finished writing to DRAM.
  // Reserving complete tile lifetimes prevents unlimited prefetching.
  std::vector<U> slots(s.buffer_slots, 0);
  // Resource calendars store non-overlapping [start, finish) intervals.
  std::map<std::string, std::map<U, U>> busy;
  U pe_done = 0, vpu_done = 0;
  // Earliest free interval on all required resources, including future reservations.
  auto event =
      [&](U tile, const std::string& phase, U ready, U duration, std::vector<std::string> resources)
  {
    U start = ready;
    if (duration)
    {
      for (;;)
      {
        U next = start;
        for (const auto& resource : resources)
        {
          auto& intervals = busy[resource];
          auto it = intervals.lower_bound(start);
          // The preceding reservation may extend across our proposed start.
          if (it != intervals.begin())
            next = std::max(next, std::prev(it)->second);
          // Later reservations can also conflict with the proposed duration.
          for (; it != intervals.end() && it->first < Add(start, duration); ++it)
            next = std::max(next, it->second);
        }
        if (next == start)
          break;
        start = next;
      }
      for (const auto& resource : resources)
        busy[resource].emplace(start, Add(start, duration));
    }
    U end = Add(start, duration);
    out.total = std::max(out.total, end);
    out.serial = Add(out.serial, duration);
    // Bound report size without truncating the schedule or its cycle totals.
    if (duration && out.trace.size() < 256)
      out.trace.push_back({tile, start, end, phase, resources});
    return end;
  };
  // Memory transfers share ports; DRAM reads and writes also share one DMA.
  // A direct VPU-to-DRAM write uses DMA but does not occupy a buffer port.
  auto transfer =
      [&](U tile, const std::string& phase, U ready, U duration, const std::string& port, bool dma)
  {
    std::vector<std::string> resources;
    if (!port.empty())
    {
      resources.push_back(port);
      if (shared_port)
        resources.push_back("buffer_shared");
    }
    if (dma)
      resources.push_back("dma");
    return event(tile, phase, ready, duration, resources);
  };
  for (U t = 0; t < costs.size(); ++t)
  {
    // Phase order matches the cost array assembled by Topology::EvaluateVPU:
    // prefetch, PE read, PE compute, PE write, VPU load, VPU compute,
    // VPU output write, and optional buffer-to-DRAM drain.
    const auto& c = costs[t];
    auto fetch = transfer(t, "prefetch", slots[t % s.buffer_slots], c[0], "buffer_write", true);
    auto read = transfer(t, "pe_read", std::max(fetch, pe_done), c[1], "buffer_read", false);
    auto pe = event(t, "pe", read, c[2], {"pe"});
    auto store = transfer(t, "pe_write", pe, c[3], "buffer_write", false);
    U load;
    if (s.input_source == "pe")
    {
      // One completed output tile can stay at the PE; no unbounded hidden FIFO.
      load = event(t, "pe_to_vpu", std::max(store, vpu_done), c[4], {"pe_output", "vpu_input"});
      // Block the next PE tile until the current result has been accepted.
      pe_done = load;
    }
    else
    {
      load = transfer(t, "vpu_read", std::max(store, vpu_done), c[4], "buffer_read", false);
      // Buffered results release the PE as soon as its buffer write completes.
      pe_done = store;
    }
    auto compute = event(t, "vpu", load, c[5], {"vpu"});
    bool direct = s.output_destination == "dram";
    vpu_done = transfer(t, direct ? "vpu_to_dram" : "vpu_write", compute, c[6],
                        direct ? "" : "buffer_write", direct);
    // Reuse this slot only after the selected output path reaches DRAM.
    slots[t % s.buffer_slots] =
        direct ? vpu_done : transfer(t, "drain", vpu_done, c[7], "buffer_read", true);
    out.compute = Add(out.compute, c[5]);
    out.write = Add(out.write, c[6]);
  }
  return out;
}
} // namespace model
