# Small MAC + VPU example

This example uses the PyTimeloop v4 frontend, matching the exercises style:

```python
import pytimeloop.timeloopfe.v4 as tl

spec = tl.Specification.from_yaml_files(ARCH_PATH, PROB_PATH, MAP_PATH)
tl.call_model(spec, output_dir=OUT_DIR)
```

## Run

From `/home/akshat/workspace/timeloop-dev`:

```bash
source .venv/bin/activate
python3 timeloop-accelergy-exercises/workspace/tutorial_exercises/vpu_intro/run.py
```

`run.sh` also forwards to this Python script. The script selects the native
`build/` executable and library. It restarts Python once before importing
PyTimeloop if necessary to avoid the virtualenv's older library copy.

Unlike the previous direct Engine runner, `tl.call_model()` processes the v4
specification and launches `timeloop-model`. Accelergy generates its ERT/ART.
Results are in `outputs/timeloop-model.stats.txt`; processed native YAML is in
`outputs/parsed-processed-input.yaml`. Old `pytimeloop.*` files are from the
previous runner and are no longer updated.

## Inputs

- `arch.yaml`: v4 `!Container` and `!Component` nodes describe DRAM, a 64-word
  shared buffer, one MAC, and one shared four-lane VPU. The embedded `vpu`
  compound definition supplies zero energy/area placeholders for Accelergy.
  Native C++ reads the VPU timing attributes. Timings are illustrative, and
  energy results are not calibrated hardware measurements.
- `problem.yaml`: A[4,4] times B[4,4] produces C[4,4]: 64 MAC operations and
  16 outputs.
- `mapping.yaml`: a fixed temporal mapping, with two completed output pieces
  of shape C[2,4]. Each needs A[2,4] + B[4,4] + C[2,4] = 32 buffer words.

| Dimension | DRAM factor | Shared-buffer factor | Total |
|---|---:|---:|---:|
| M | 2 | 2 | 4 |
| N | 1 | 4 | 4 |
| K | 1 | 4 | 4 |

Permutation `[K, N, M]` is inner-to-outer. All K iterations are inside the
shared-buffer tile, so each piece contains completed outputs. There is one
MAC, so no spatial mapping is needed.

## Connected serial execution

`problem.yaml` now requests ReLU on C through `vpu_stages`. `arch.yaml` names
`shared_buffer` and `DRAM` as the VPU connections. The existing mapping already
completes the K reduction inside each eight-output piece, so its factors remain
unchanged.

For each piece, native `Topology::EvaluateVPU()` accounts for:

```
input prefetch -> producer buffer read -> MAC computation -> producer buffer write
    -> VPU buffer read -> VPU computation -> direct DRAM write
```

The next piece starts after this one finishes. There is no overlap. The VPU is
called once to obtain the cost of the identical-sized pieces, and that cost is
charged for every piece. No tensor values are calculated by this timing model.

The default hand calculation across both pieces is:

| Phase | Cycles |
|---|---:|
| Input prefetch | 8 |
| Producer buffer reads | 44 |
| MAC computation | 64 |
| Producer buffer writes | 16 |
| VPU buffer reads | 4 |
| VPU ReLU | 4 |
| Final DRAM writes | 4 |
| Total | 144 |

The baseline 64-cycle estimate is printed only for comparison. It is not added
again: the serial phases replace its overlapped MAC/memory timing. This also
means the difference from 64 cycles is not solely ReLU overhead. Original final
output traffic is moved after the VPU, not duplicated. Fixed network latency,
if configured, is added once after the phase total.

Producer traffic and cycles come from Timeloop's aggregate mapping statistics
and are divided evenly across tiles, preserving totals. This is a conservative
analytical model, not a reconstruction of the exact MAC event timeline. Native
memory energy counters remain the original baseline, including its leakage
time; additional VPU energy and buffer-read energy are not modeled.

This first connection supports one in-place activation (ReLU, sigmoid, tanh,
or softmax). Softmax requires `vector_length` to fit within the tile's final
output axis, and its configured latency describes a complete vector. Binary
operations remain available in the standalone VPU class, but connected binary
operands are rejected until input-tensor handling is implemented. Explicit
layout modeling, sparse memory, and partial reductions outside the shared
buffer are rejected.

## Latency versus initiation interval

`latency` is the time from starting a vector batch until its result is ready.
`initiation_interval` is the minimum spacing between starting two batches.

For 8 elements and vector_width=4, there are two batches:

- latency=4, initiation_interval=4: start at 0 and 4; finish at 4 and 8.
- latency=4, initiation_interval=1: start at 0 and 1; finish at 4 and 5.

Thus `cycles = latency + (batches - 1) * initiation_interval` for nonempty work.
The interval defaults to latency, giving serial batches. Allowing overlap
between batches within one VPU operation does not overlap the MAC/VPU/memory
phases or different producer pieces.

## Verify

After running the example:

```bash
python3 timeloop-accelergy-exercises/workspace/tutorial_exercises/vpu_intro/test_connected_vpu.py
```

The tests check the hand-calculated total, width/latency/interval changes, the
64-cycle baseline when the stage is removed, and rejection of unsupported or
incomplete work. Inspect the `VPU serial flow` section of the stats report for
the phase breakdown. YAML timing changes only require rerunning. Native/header
changes require rebuilding Timeloop and matching Python bindings.
