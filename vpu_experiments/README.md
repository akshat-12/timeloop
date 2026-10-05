# Single-problem VPU experiments

`problem.yaml` describes A[1024,1024] x B[1024,1024] -> C[1024,1024].
Edit its M/N/K instance to change the problem; M and N must be multiples of 64.
The previous five-problem collection has been removed.

`arch.yaml` defines 64 PEs, 16-bit words, a 2 MiB shared buffer, two tile slots,
and buffer/DRAM bandwidths of 64/32 words per cycle. Completed output tiles are
64x64, with all K reduction steps inside each tile. If you increase the problem,
ensure the buffer fits two complete tile working sets and DRAM fits all tensors.

The four files in `mappings/` select:

- PE -> VPU -> DRAM
- PE -> VPU -> SharedBuf -> DRAM
- PE -> SharedBuf -> VPU -> DRAM
- PE -> SharedBuf -> VPU -> SharedBuf -> DRAM

Their loop mapping is identical. `vpu_route` is metadata consumed by the script,
which applies it to the architecture and strips it before invoking PyTimeloop.
The script adjusts loop factors to match `problem.yaml`.

## Run

From `/home/akshat/workspace/timeloop-dev`:

```bash
source .venv/bin/activate
python3 accelergy-timeloop-infrastructure/src/timeloop/vpu_experiments/run_experiments.py
```

Three sweeps run on this one problem:

1. ReLU, sigmoid, tanh at width 64 with buffered input/output. Latencies are
   configured in `arch.yaml` (currently 8, 16, 32 cycles per batch).
2. All four routes with ReLU and width 64.
3. Widths 8, 16, 32, 64, 128, 256 with ReLU and buffered input/output. Both PE
   output width and VPU input width change together; PE count stays fixed.

There are 13 comparison rows and 11 distinct simulations because repeated
baseline combinations are reused. Other hardware and mapping choices stay fixed.
All plots are **line graphs with point markers and exact cycle annotations**.
Categorical activation/route points are joined for visual comparison, not to
imply interpolation. Y-axis ranges are zoomed and labelled to expose differences.

## Results

- `results/activations.csv` and `results/plots/activations.png` / `.svg`
- `results/routes.csv` and `results/plots/routes.png` / `.svg`
- `results/widths.csv` and `results/plots/widths.png` / `.svg`
- `results/runs.json`: all distinct simulation results.
- `results/runs/`: exact YAML inputs, logs and native outputs for each run.

Each plot shows only total latency in cycles. VPU compute service time is
already included in total latency. Results are analytical estimates, not hardware measurements. The greedy
scheduler can produce non-monotonic latency, and energy is not compared.

From this directory, after activating the virtualenv:

```bash
python3 run_experiments.py --experiment activations
python3 run_experiments.py --experiment routes
python3 run_experiments.py --experiment widths --widths 16 32 64 128
/usr/bin/python3 run_experiments.py --plot-only
```

Use `--results path` for a separate experiment, `--timeout 180` for the per-run
limit, or `--no-plots` for CSVs only. Re-running replaces selected outputs and
sweep CSVs. After editing the problem, run all sweeps to replace all old CSVs.

PyTimeloop runs in the virtualenv. The script uses system Python for plotting
when that virtualenv lacks matplotlib; override with `--plot-python /path/to/python`.
Accelergy tables are reused per width for fixed hardware and cleared each run.
Timing is still simulated separately for every distinct configuration.
