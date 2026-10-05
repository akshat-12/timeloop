#!/usr/bin/env python3
"""Run three controlled VPU sweeps and plot total latency in cycles."""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "build"
ACTIVATIONS = ("relu", "sigmoid", "tanh")
ROUTES = (
    "pe_vpu_dram", "pe_vpu_buffer_dram",
    "pe_buffer_vpu_dram", "pe_buffer_vpu_buffer_dram",
)
ROUTE_LABELS = (
    "PE → VPU → DRAM", "PE → VPU → Buffer → DRAM",
    "PE → Buffer → VPU → DRAM", "PE → Buffer → VPU → Buffer → DRAM",
)
BASE_ROUTE = "pe_buffer_vpu_buffer_dram"
BASE_WIDTH = 64
TILE_M = TILE_N = 64

def load_yaml(path):
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True

    return yaml.load(path)

def save_yaml(path, value):
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True

    with path.open("w") as stream:
        yaml.dump(value, stream)

def worker(directory):
    """Keep frontend and native global state isolated in a fresh process."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    import pytimeloop.timeloopfe.v4 as tl

    spec = tl.Specification.from_yaml_files(
        str(directory / "arch.yaml"), str(directory / "problem.yaml"),
        str(directory / "mapping.yaml"),
    )

    tables = [str(directory / name) for name in ("ERT.yaml", "ART.yaml")
              if (directory / name).exists()]

    # The frontend also reads ART from its output directory after native execution.
    # Native Timeloop does not re-emit tables supplied as input, so retain copies.
    (directory / "outputs").mkdir(exist_ok=True)
    for table in tables:
        shutil.copy2(table, directory / "outputs" / f"timeloop-model.{Path(table).name}")

    tl.call_model(spec, output_dir=str(directory / "outputs"), extra_input_files=tables)

def metric(text, label):
    match = re.search(r"^  " + re.escape(label) + r": (\d+)", text, re.M)
    if match is None:
        raise RuntimeError(f"Missing native statistic: {label}")

    return int(match[1])

def simulate(problem_path, operation, route_name, width, results, timeout):
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString as Quoted

    directory = results / "runs" / problem_path.stem / f"{operation}__{route_name}__w{width}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "outputs").exists():
        shutil.rmtree(directory / "outputs")

    architecture = load_yaml(HERE / "arch.yaml")
    problem = load_yaml(problem_path)
    mapping = load_yaml(HERE / "mappings" / f"{route_name}.yaml")
    route = mapping.pop("vpu_route")

    nodes = {node["name"]: node for node in architecture["architecture"]["nodes"]}
    vpu = nodes["VPU"]["attributes"]
    vpu["input_source"] = Quoted(route["input_source"])
    vpu["output_destination"] = Quoted(route["output_destination"])
    vpu["input_width"] = nodes["mac"]["attributes"]["output_width"] = width

    problem["problem"]["vpu_stages"][0]["operation"] = operation

    dimensions = problem["problem"]["instance"]
    m, n, k = (int(dimensions[d]) for d in "MNK")
    if m % TILE_M or n % TILE_N:
        raise ValueError("Problem M and N must be multiples of 64 for this mapping")

    for entry in mapping["mapping"]:
        if entry["type"] == "temporal":
            if entry["target"] == "DRAM":
                factors = (m // TILE_M, n // TILE_N, 1)
            else:
                factors = (TILE_M // 8, TILE_N // 8, k)
            entry["factors"] = [f"{d}={value}" for d, value in zip("MNK", factors)]

    for filename, data in (("arch.yaml", architecture), ("problem.yaml", problem),
                           ("mapping.yaml", mapping)):
        save_yaml(directory / filename, data)

    # Memory/PE hardware is fixed across problems and activations. Reuse its
    # Accelergy tables at each width; route changes only affect the timing model.
    # Tables are regenerated at the start of each invocation, so edits to arch.yaml
    # cannot accidentally use energy tables from a previous experiment.
    table_cache = results / "energy_tables" / f"width_{width}"
    for name in ("ERT.yaml", "ART.yaml"):
        (directory / name).unlink(missing_ok=True)
        if (table_cache / name).exists():
            shutil.copy2(table_cache / name, directory / name)

    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(directory)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError(f"Timeout: inspect {directory / 'run.log'}") from None

    if code:
        raise RuntimeError(f"Model exited with {code}: inspect {directory / 'run.log'}")

    table_cache.mkdir(parents=True, exist_ok=True)
    for name in ("ERT.yaml", "ART.yaml"):
        generated = directory / "outputs" / f"timeloop-model.{name}"
        if generated.exists():
            shutil.copy2(generated, table_cache / name)

    text = (directory / "outputs/timeloop-model.stats.txt").read_text()
    cycles = int(re.search(r"^Cycles: (\d+)", text, re.M)[1])
    tiles = (m // TILE_M) * (n // TILE_N)
    batches = (TILE_M * TILE_N + width - 1) // width
    latency = int(vpu["operations"][operation]["latency"])

    # Validate exact workload/batch counts and route-specific final-output traffic.
    expected = {
        "Completed tiles": tiles, "Batches per tile": batches,
        "Matched interface width": width, "VPU compute cycles": tiles * batches * latency,
        "PE final buffer write words": 0 if route["input_source"] == "pe" else m*n,
        "VPU buffer read words": 0 if route["input_source"] == "pe" else m*n,
        "VPU buffer write words": 0 if route["output_destination"] == "dram" else m*n,
        "Final buffer drain read words": 0 if route["output_destination"] == "dram" else m*n,
        "Final DRAM write words": m*n,
    }

    for label, value in expected.items():
        if metric(text, label) != value:
            raise RuntimeError(f"Incorrect {label} in {directory}")

    if cycles != metric(text, "Pipeline cycles"):
        raise RuntimeError("Unexpected network latency; revisit this experiment's assumptions")

    if cycles > metric(text, "Same-work serial cycles"):
        raise RuntimeError("Pipeline exceeds serial service time")

    return dict(problem=problem_path.stem, M=m, N=n, K=k, activation=operation,
                route=route_name, input_width=width, latency_per_batch=latency,
                cycles=cycles, vpu_compute_cycles=metric(text, "VPU compute cycles"),
                serial_cycles=metric(text, "Same-work serial cycles"),
                tiles=tiles, batches_per_tile=batches, run_directory=str(directory))

def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def plots(results):
    """Plot saved CSVs; this mode does not import PyTimeloop or require ruamel."""
    os.environ.setdefault("MPLCONFIGDIR", str(results / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})

    plot_dir = results / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for experiment in ("activations", "routes", "widths"):
        csv_path = results / f"{experiment}.csv"
        if not csv_path.exists():
            continue

        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))

        if len({row["problem"] for row in rows}) != 1:
            raise ValueError("Results contain multiple problems; rerun the single-problem sweeps")

        figure, ax = plt.subplots(figsize=(13, 5), layout="constrained")
        shape = f"M={rows[0]['M']}, N={rows[0]['N']}, K={rows[0]['K']}"
        values = [int(row["cycles"]) for row in rows]

        if experiment == "widths":
            x = [int(row["input_width"]) for row in rows]
            labels = list(map(str, x))
            ax.set_xscale("log", base=2)
            xlabel = "Matched PE output / VPU input width (elements)"
        else:
            x = list(range(len(rows)))
            labels = ([ROUTE_LABELS[ROUTES.index(row["route"])].replace(" → ", "\n→ ")
                       for row in rows] if experiment == "routes"
                      else [row["activation"] for row in rows])
            xlabel = "Completed-output route" if experiment == "routes" else "Activation"

        ax.plot(x, values, "o-", color="#2563eb", linewidth=2, markersize=7)
        ax.set_xticks(x, labels=labels)
        ax.set_xlabel(xlabel)

        for position, value in zip(x, values):
            ax.annotate(f"{value:,}", (position, value), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=9)

        ax.set_title(f"{shape} — Total latency", fontsize=11)
        ax.set_ylabel("Cycles")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

        # A clearly labelled, nonzero range makes small latency differences visible.
        span = max(values) - min(values)
        padding = max(span * 0.25, max(values) * 0.001, 1)
        ax.set_ylim(max(0, min(values) - padding), max(values) + padding)
        ax.margins(x=0.12)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

        subtitles = {
            "activations": "Activation sweep · buffered input/output · width 64",
            "routes": "Route comparison · ReLU · width 64",
            "widths": "Interface-width sweep · ReLU · buffered input/output",
        }
        figure.suptitle(subtitles[experiment] + "\nAnalytical model; all other hardware and tiling fixed", fontsize=14)

        for extension in ("png", "svg"):
            figure.savefig(plot_dir / f"{experiment}.{extension}", dpi=160)

        plt.close(figure)
        print(f"Plot: {plot_dir / (experiment + '.png')}", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "activations", "routes", "widths"), default="all")
    parser.add_argument("--widths", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--timeout", type=float, default=180, help="Seconds per native run")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Plot existing CSVs without simulations")
    parser.add_argument("--plot-python", help="Interpreter with matplotlib, if different from this Python")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker:
        worker(args.worker)
        return

    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        plots(results)
        return

    if args.timeout <= 0 or any(width <= 0 for width in args.widths):
        parser.error("Timeout and widths must be positive")

    if len(args.widths) != len(set(args.widths)):
        parser.error("Widths must be distinct")

    problem = HERE / "problem.yaml"
    for width in set(args.widths) | {BASE_WIDTH}:
        for name in ("ERT.yaml", "ART.yaml"):
            (results / "energy_tables" / f"width_{width}" / name).unlink(missing_ok=True)

    cache = {}

    def measure(problem, activation, route, width):
        key = (problem.stem, activation, route, width)
        if key not in cache:
            print(f"Running {problem.stem}: {activation}, {route}, width={width}", flush=True)
            cache[key] = simulate(problem, activation, route, width, results, args.timeout)
            print(f"  cycles={cache[key]['cycles']:,}", flush=True)

        return cache[key]

    # Sweep 1: change only activation; reuse the same problem, mapping and width.
    if args.experiment in ("all", "activations"):
        rows = []
        for activation in ACTIVATIONS:
            rows.append(measure(problem, activation, BASE_ROUTE, BASE_WIDTH))
            write_csv(results / "activations.csv", rows)

    # Sweep 2: change only routing; all four run the identical matmul and ReLU.
    if args.experiment in ("all", "routes"):
        rows = []
        for route in ROUTES:
            rows.append(measure(problem, "relu", route, BASE_WIDTH))
            write_csv(results / "routes.csv", rows)

    # Sweep 3: change both matched interfaces together, holding PE count fixed.
    if args.experiment in ("all", "widths"):
        rows = []
        for width in sorted(args.widths):
            rows.append(measure(problem, "relu", BASE_ROUTE, width))
            write_csv(results / "widths.csv", rows)

    (results / "runs.json").write_text(json.dumps(list(cache.values()), indent=2) + "\n")
    print(f"Completed {len(cache)} distinct simulations. CSVs: {results}", flush=True)

    if not args.no_plots:
        plot_python = args.plot_python or sys.executable
        if not args.plot_python and importlib.util.find_spec("matplotlib") is None:
            # This workspace's system Python has matplotlib; its virtualenv has PyTimeloop.
            plot_python = "/usr/bin/python3"

        subprocess.run([plot_python, str(Path(__file__).resolve()), "--plot-only",
                        "--results", str(results)], check=True)

if __name__ == "__main__":
    # The native extension and executable must load the same rebuilt shared library.
    os.environ["PATH"] = str(BUILD) + os.pathsep + os.environ.get("PATH", "")

    if os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)[0] != str(BUILD):
        os.environ["LD_LIBRARY_PATH"] = str(BUILD) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])

    main()
