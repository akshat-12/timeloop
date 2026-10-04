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

Read `outputs/timeloop-model.stats.txt`. Default total latency is 88 cycles,
versus 112 for the same work serialized. PE tile 1 computes at cycles 50-58;
VPU tile 0 computes at 36-52. They overlap at 50-52.

## Parameters

In `arch.yaml`, change VPU `input_width` and MAC `output_width` together; both
are four elements by default. Set each operation's `latency` in the VPU's
`operations` map. Latency is per batch; eight outputs at width four and ReLU
latency eight take two batches, or 16 compute cycles per tile. Transfer cycles
are separate. Two buffer slots each reserve a full 32-word tile working set.

| VPU input_source | VPU output_destination | Expected cycles |
|---|---|---:|
| `pe` | `dram` | 82 |
| `pe` | `shared_buffer` | 84 |
| `shared_buffer` | `dram` | 86 |
| `shared_buffer` | `shared_buffer` | 88 |

Quote these route values in YAML. The route switches apply to completed outputs;
earlier partial-sum traffic remains included. YAML changes need no rebuild.
VPU energy/area and route-adjusted memory energy are not modeled.
