#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import torch


DEFAULT_RESULTS_ROOT = "results"
DEFAULT_OUT_DIR = "latex_tables"

PREFERRED_DATASET_ORDER = [
    "gmm2d",
    "gmmhd",
    "gmmhd_anisotropic",
    "joker",
    "himmelblau",
    "funnel2d",
    "funnel",
]

PREFERRED_ALGO_ORDER = [
    "a-SVGD",
    "SVGD",
    "ALDI",
    "MSIP-Fredholm",
    "MSIP-GI-1",
    "MSIP-GI-10",
    "GA",
    "MSIP-GF",
    "CBS",
]

#    "a-SVGD",

ALGO_DISPLAY = {
    "a-SVGD": "a-SVGD",
    "SVGD": "SVGD",
    "ALDI": "ALDI",
    "GI-ALDI": "ALDI",          # backward compatibility
    "MSIP-Fredholm": r"\textbf{MSIPF}",
    "MSIP-GI-1": r"\textbf{MSIPGI-1}",
    "MSIP-GI-10": r"\textbf{MSIPGI-10}",
    "GA": "GA",
    "DeepEnsembles": "GA",      # backward compatibility
    "MSIP-GF": r"\textbf{MSIPGF}",
    "CBS": "CBS",
}

GRADIENT_INFORMED = [
    "a-SVGD",
    "SVGD",
    "ALDI",
    "GI-ALDI",          # backward compatibility
    "MSIP-Fredholm",
    "MSIP-GI-1",
    "MSIP-GI-10",
    "GA",
    "DeepEnsembles",    # backward compatibility
]

GRADIENT_FREE = ["MSIP-GF", "CBS"]

DATASET_DISPLAY_OVERRIDES = {
    "gmm2d": "GMM2",
    "joker": "Joker",
    "himmelblau": "Himmelblau",
    "funnel": "Funnel",
    "funnel2d": "Funnel2",
    "gmmhd": "GMM",
    "gmmhd_anisotropic": "GMM",
}


# =============================================================================
# Loading helpers
# =============================================================================

def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def canonical_metric_name(name: str) -> str:
    name = str(name).strip()

    name = name.replace("cross_entropy", "CE")
    name = name.replace("CrossEntropy", "CE")

    parts = name.split("_")
    fixed = []

    for p in parts:
        low = p.lower()

        if low == "ksd":
            fixed.append("KSD")
        elif low == "rbf":
            fixed.append("RBF")
        elif low == "imq":
            fixed.append("IMQ")
        elif low == "ce":
            fixed.append("CE")
        elif low == "weighted":
            fixed.append("weighted")
        elif low == "unweighted":
            fixed.append("unweighted")
        else:
            fixed.append(p)

    return "_".join(fixed)


def canonical_experiment_name(exp_name: str) -> str:
    name = exp_name.strip()
    name = re.sub(r"_+$", "", name)
    name = name.replace("__", "_")
    return name


def canonical_algo_name(algo_name: str) -> str:
    """Normalize old algorithm names to the current table names."""
    algo_name = str(algo_name)

    aliases = {
        "GI-ALDI": "ALDI",
        "DeepEnsembles": "GA",
    }

    return aliases.get(algo_name, algo_name)


def infer_dataset_and_dim(exp_name: str, payload: dict[str, Any] | None = None):
    name = canonical_experiment_name(exp_name)

    if payload is not None:
        for key in ["DIM", "dim", "d"]:
            if key in payload:
                try:
                    return name, int(payload[key])
                except Exception:
                    pass

    if name.startswith("gmmhd_anisotropic"):
        return "gmmhd_anisotropic", None
    if name.startswith("gmmhd"):
        return "gmmhd", None
    if name.startswith("gmm2d"):
        return "gmm2d", 2
    if name.startswith("joker"):
        return "joker", 2
    if name.startswith("himmelblau"):
        return "himmelblau", 2
    if name.startswith("funnel2d"):
        return "funnel2d", 2
    if name.startswith("funnel_5d") or name.startswith("funnel_ksd_5d"):
        return "funnel", 5
    if name.startswith("funnel"):
        return "funnel", 2

    m = re.search(r"(?<!\d)(\d+)d(?!\d)", name)
    if m:
        clean_name = re.sub(r"_?\d+d", "", name)
        return clean_name, int(m.group(1))

    return name, None


def dataset_display_name(dataset: str, dim: int | None) -> str:
    base = DATASET_DISPLAY_OVERRIDES.get(dataset, dataset)

    if dataset.startswith("gmmhd"):
        return f"GMM{dim}" if dim is not None else "GMM"

    if dataset.startswith("funnel") and dim is not None and dim != 2:
        return f"Funnel{dim}"

    return base


def metric_files_for_experiment(exp_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    preferred_dirs = [
        exp_dir / "metrics",
        exp_dir / "metrics" / "raw_values",
        exp_dir / "metrics" / "summary_values",
        exp_dir / "summaries",
    ]

    for d in preferred_dirs:
        if d.exists():
            candidates.extend(sorted(d.glob("*.pt")))

    if candidates:
        return sorted(set(candidates))

    excluded_parts = {
        "particles",
        "weights",
        "combined_particles_weights",
        "final_states",
        "plots",
    }

    for p in exp_dir.rglob("*.pt"):
        if any(part in excluded_parts for part in p.parts):
            continue
        candidates.append(p)

    return sorted(set(candidates))


def metric_name_from_file(path: Path, payload: dict[str, Any] | None = None) -> str:
    if payload is not None:
        for key in ["metric_name", "metric"]:
            if key in payload:
                return canonical_metric_name(str(payload[key]))

    stem = path.stem

    stem = re.sub(r"^.*?(cross_entropy|CE|ksd|KSD)", r"\1", stem)
    stem = re.sub(r"_summary$", "", stem)
    stem = re.sub(r"_M_.*$", "", stem)
    stem = re.sub(r"_D_.*$", "", stem)
    stem = re.sub(r"_T_.*$", "", stem)

    return canonical_metric_name(stem)


def parse_dim_M_key(key):
    dim = None
    M = None

    if isinstance(key, tuple) and len(key) == 2:
        dim, M = key

    elif isinstance(key, int):
        M = key

    elif isinstance(key, str):
        m_tuple = re.match(r"^\(?\s*(\d+)\s*,\s*(\d+)\s*\)?$", key)
        if m_tuple:
            dim = int(m_tuple.group(1))
            M = int(m_tuple.group(2))
        elif key.isdigit():
            M = int(key)

    return dim, M


def extract_metric_results_from_payload(
    payload: Any,
    fallback_metric_name: str | None = None,
) -> dict[str, dict[tuple[int | None, int | None, str], list[float]]]:
    """
    Return:

        {
            metric_name: {
                (dim, M, algo): [values over runs]
            }
        }

    Handles:

        payload["values"][M][algo] = [values]

        payload["results"][M][algo] = [values]

        payload["results"][(d, M)][algo] = [values]

        payload["results"][M][algo][metric_name] = [values]

        payload["results"][(d, M)][algo][metric_name] = [values]
    """
    if not isinstance(payload, dict):
        return {}

    raw = (
        payload.get("values")
        or payload.get("results")
        or payload.get("metric_values")
    )

    if not isinstance(raw, dict):
        return {}

    out: dict[str, dict[tuple[int | None, int | None, str], list[float]]] = {}

    for key, algo_dict in raw.items():
        dim, M = parse_dim_M_key(key)

        if not isinstance(algo_dict, dict):
            continue

        for algo_name, vals in algo_dict.items():

            # Dedicated metric file:
            # results[M][algo] = [values]
            if isinstance(vals, (list, tuple)):
                if fallback_metric_name is None:
                    continue

                metric_name = canonical_metric_name(fallback_metric_name)
                out.setdefault(metric_name, {})
                out[metric_name][(dim, M, canonical_algo_name(str(algo_name)))] = [float(v) for v in vals]

            # Combined metric file:
            # results[M][algo][metric_name] = [values]
            elif isinstance(vals, dict):
                for metric_name_raw, metric_vals in vals.items():
                    if not isinstance(metric_vals, (list, tuple)):
                        continue

                    metric_name = canonical_metric_name(metric_name_raw)
                    out.setdefault(metric_name, {})
                    out[metric_name][(dim, M, canonical_algo_name(str(algo_name)))] = [
                        float(v) for v in metric_vals
                    ]

    return out


def load_all_metric_results(results_root: Path):
    all_results: dict[str, list[dict[str, Any]]] = {}

    for exp_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        files = metric_files_for_experiment(exp_dir)

        for f in files:
            payload = safe_torch_load(f)

            if not isinstance(payload, dict):
                continue

            fallback_metric_name = metric_name_from_file(f, payload)
            dataset, default_dim = infer_dataset_and_dim(exp_dir.name, payload)

            metric_results_by_name = extract_metric_results_from_payload(
                payload,
                fallback_metric_name=fallback_metric_name,
            )

            for metric_name, metric_results in metric_results_by_name.items():
                for (dim, M, algo), values in metric_results.items():
                    row_dim = dim if dim is not None else default_dim

                    all_results.setdefault(metric_name, []).append(
                        {
                            "dataset": dataset,
                            "dim": row_dim,
                            "M": M,
                            "algo": canonical_algo_name(algo),
                            "values": values,
                            "source": f,
                        }
                    )

    return all_results


# =============================================================================
# Formatting
# =============================================================================

def mean_std(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    mean = t.mean().item()
    std = t.std(unbiased=True).item() if len(values) > 1 else 0.0
    return mean, std


def fmt_number(x: float, precision: int = 3) -> str:
    if not math.isfinite(x):
        return "---"

    ax = abs(x)

    if ax == 0:
        return "0"

    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{precision}e}"

    if ax >= 100:
        return f"{x:.1f}"

    if ax >= 10:
        return f"{x:.2f}"

    return f"{x:.{precision}f}"


def fmt_mean_std(mean: float, std: float, bold: bool = False, precision: int = 3) -> str:
    text = f"{fmt_number(mean, precision)} ({fmt_number(std, precision)})"
    if bold:
        return r"\textbf{" + text + "}"
    return text


def metric_display_name(metric_name: str) -> str:
    return metric_name.replace("_", r"\_")


def metric_filename_safe(metric_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", metric_name)


def collect_algorithms(rows: list[dict[str, Any]]) -> list[str]:
    found = sorted({r["algo"] for r in rows})
    ordered = [a for a in PREFERRED_ALGO_ORDER if a in found]
    ordered += [a for a in found if a not in ordered]
    return ordered


def sort_dataset_rows(rows: list[dict[str, Any]]):
    keys = sorted({(r["dataset"], r["dim"], r["M"]) for r in rows})

    def key_fn(k):
        dataset, dim, M = k
        try:
            pref = PREFERRED_DATASET_ORDER.index(dataset)
        except ValueError:
            pref = 999

        return (
            pref,
            dataset,
            dim if dim is not None else 10**9,
            M if M is not None else 10**9,
        )

    return sorted(keys, key=key_fn)


def make_latex_table(
    metric_name: str,
    rows: list[dict[str, Any]],
    caption: str | None = None,
    label: str | None = None,
    precision: int = 3,
) -> str:
    algorithms = collect_algorithms(rows)

    gi_algos = [a for a in algorithms if a in GRADIENT_INFORMED]
    gf_algos = [a for a in algorithms if a in GRADIENT_FREE]
    other_algos = [a for a in algorithms if a not in gi_algos and a not in gf_algos]

    gi_algos += other_algos
    algorithms = gi_algos + gf_algos

    n_algo = len(algorithms)
    n_gi = len(gi_algos)
    n_gf = len(gf_algos)

    if caption is None:
        caption = (
            "Examining different algorithms for several inference tasks. "
            f"All quantification is done using {metric_display_name(metric_name)}. "
            "Entries report mean and standard deviation over independent runs; "
            "the smallest mean in each row is highlighted in bold."
        )

    if label is None:
        label = "tab:" + metric_filename_safe(metric_name).lower()

    col_spec = "@{}l c " + " ".join(["c"] * n_algo) + "@{}"

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"{")
    lines.append(r"    \caption{" + caption + r"}")
    lines.append(r"    \tiny")
    lines.append(r"    \centering")
    lines.append(r"    \label{" + label + r"}")
    lines.append(r"    \begin{tabular}{" + col_spec + r"}\toprule")

    if n_gi > 0 and n_gf > 0:
        lines.append(
            "    & & "
            + rf"\multicolumn{{{n_gi}}}{{c}}{{Gradient-informed algorithms}}"
            + " & "
            + rf"\multicolumn{{{n_gf}}}{{c}}{{Gradient-free algorithms}}"
            + r" \\"
        )
        lines.append(
            rf"    \cmidrule(lr){{3-{2+n_gi}}}"
            + rf"\cmidrule(lr){{{3+n_gi}-{2+n_gi+n_gf}}}"
        )
    elif n_gi > 0:
        lines.append(
            "    & & "
            + rf"\multicolumn{{{n_gi}}}{{c}}{{Algorithms}} \\"
        )
        lines.append(rf"    \cmidrule(lr){{3-{2+n_gi}}}")

    header = "     & Dim."
    for algo in algorithms:
        header += " & " + ALGO_DISPLAY.get(algo, algo)
    header += r" \\\midrule"
    lines.append(header)

    for dataset, dim, M in sort_dataset_rows(rows):
        algo_to_vals = {}

        for r in rows:
            if r["dataset"] == dataset and r["dim"] == dim and r["M"] == M:
                algo_to_vals[r["algo"]] = r["values"]

        means = {}
        stds = {}

        for algo, vals in algo_to_vals.items():
            means[algo], stds[algo] = mean_std(vals)

        available_means = [
            means[a]
            for a in algorithms
            if a in means and math.isfinite(means[a])
        ]

        best = min(available_means) if available_means else None

        row_label = dataset_display_name(dataset, dim)
        dim_text = "---" if dim is None else str(dim)

        line = f"    {row_label} & {dim_text}"

        for algo in algorithms:
            if algo not in means:
                line += " & ---"
            else:
                is_best = (
                    best is not None
                    and math.isclose(means[algo], best, rel_tol=1e-12, abs_tol=1e-12)
                )
                line += " & " + fmt_mean_std(
                    means[algo],
                    stds[algo],
                    bold=is_best,
                    precision=precision,
                )

        line += r" \\"
        lines.append(line)

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(DEFAULT_RESULTS_ROOT),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(DEFAULT_OUT_DIR),
    )

    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric names to export. If omitted, export all detected metrics.",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=3,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {args.results_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_all_metric_results(args.results_root)

    if not all_results:
        raise RuntimeError(
            f"No metric results found under {args.results_root}. "
            "Expected files like results/EXP_NAME/metrics/*.pt."
        )

    metric_names = sorted(all_results)

    if args.metrics:
        requested = {canonical_metric_name(m) for m in args.metrics}
        metric_names = [m for m in metric_names if m in requested]

        missing = sorted(requested - set(metric_names))
        if missing:
            print("Warning: requested metrics not found:")
            for m in missing:
                print(f"  - {m}")

    all_tables = []

    for metric_name in metric_names:
        table = make_latex_table(
            metric_name,
            all_results[metric_name],
            precision=args.precision,
        )

        out_path = args.out_dir / f"table_{metric_filename_safe(metric_name)}.tex"
        out_path.write_text(table + "\n", encoding="utf-8")

        all_tables.append(table)

        print(f"Wrote {out_path}")

    combined_path = args.out_dir / "all_metric_tables.tex"
    combined_path.write_text("\n\n".join(all_tables) + "\n", encoding="utf-8")

    print(f"Wrote {combined_path}")

    print()
    print("Detected metrics:")
    for m in metric_names:
        print(f"  - {m}")


if __name__ == "__main__":
    main()