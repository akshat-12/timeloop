# Matmul and VPU sample

The inputs in this folder are self-contained; nothing references the exercises
repository. They use PyTimeloop v4 YAML, so use the supplied Python runner.

- `arch.yaml`: DRAM, a 64-word shared buffer, four PEs, and one shared VPU.
- `problem.yaml`: A[4,4] x B[4,4] -> C[4,4], followed by ReLU.
- `mapping.yaml`: two C[2,4] tiles, with the complete K reduction inside each tile.
- `run.py`: loads these inputs and invokes Timeloop from the adjacent `build/`.

From `/home/akshat/workspace/timeloop-dev`:

```bash
source .venv/bin/activate
python3 accelergy-timeloop-infrastructure/src/timeloop/vpu_sample/run.py
```

Read `outputs/timeloop-model.stats.txt`. Default total latency is 32 cycles,
versus 28 without the VPU stage. Timing is the maximum of PE compute, adjusted
memory traffic costs, and the five route/compute terms. Transfers and compute
are assumed to overlap; startup, drain latency and buffer-slot stalls are not modeled.

## Parameters

In `arch.yaml`, change VPU `input_width` and MAC `output_width` together; both
are four elements by default. Set each operation's `latency` in the VPU's
`operations` map. Latency is per batch; eight outputs at width four and ReLU
latency eight take two batches, or 16 compute cycles per tile. Transfer costs
are separate bounds. Traffic sharing a memory port is combined before computing
its bandwidth bound.

| VPU input_source | VPU output_destination | Expected cycles |
|---|---|---:|
| `pe` | `dram` | 32 |
| `pe` | `shared_buffer` | 32 |
| `shared_buffer` | `dram` | 32 |
| `shared_buffer` | `shared_buffer` | 32 |

Quote these route values in YAML. The route switches apply to completed outputs;
earlier partial-sum traffic remains included. YAML changes need no rebuild.
VPU energy/area and route-adjusted memory energy are not modeled.
