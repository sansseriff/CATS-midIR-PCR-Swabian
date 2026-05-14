import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_interarrival_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("plot_mode") != "special_correlation":
        raise ValueError(
            f"Expected special_correlation JSON, got plot_mode={data.get('plot_mode')!r}"
        )
    if data.get("measurement_class") != "StartStop":
        raise ValueError(
            f"Expected StartStop JSON, got measurement_class={data.get('measurement_class')!r}"
        )
    return data


def choose_y_data(data: dict, semilog: bool) -> np.ndarray:
    if semilog and "display_counts_semilog" in data:
        return np.asarray(data["display_counts_semilog"], dtype=float)

    counts = np.asarray(data["display_counts"], dtype=float)
    if semilog:
        counts = counts.copy()
        counts[counts <= 0] = np.nan
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and plot saved interarrival histogram JSON from special correlation mode."
    )
    #parser.add_argument("json_file", help="Path to the saved special-correlation JSON file")
    parser.add_argument(
        "--ps",
        action="store_true",
        help="Plot the x-axis in ps instead of ns",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear y-scale instead of semilog-y",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also overlay the raw sparse non-empty StartStop bins",
    )
    args = parser.parse_args()

    # file_path = Path(args.json_file).expanduser().resolve()

    file_path = "/home/cats/Documents/measurements/QCL_testing/4.3.2026/SECTIONHIST__QCL_63_8.39_64.3_500_80__FRIDGE_263__SAVE_180_0.012_DETECTOR_0.08__WINDOW_0.050_0.480.json"
    data = load_interarrival_json(file_path)

    x_ps = np.asarray(data["display_x_axis_ps"], dtype=float)
    y = choose_y_data(data, semilog=not args.linear)

    x_scale = 1.0 if args.ps else 1e-3
    x_unit = "ps" if args.ps else "ns"
    x = x_ps * x_scale

    fig, ax = plt.subplots()
    ax.plot(x, y, drawstyle="steps-mid", linewidth=1.2, label="Dense display histogram")

    if args.show_raw:
        raw_x_ps = np.asarray(data.get("raw_sparse_time_ps", []), dtype=float)
        raw_counts = np.asarray(data.get("raw_sparse_counts", []), dtype=float)
        if raw_x_ps.size and raw_counts.size:
            ax.scatter(
                raw_x_ps * x_scale,
                raw_counts,
                s=14,
                alpha=0.7,
                label="Raw non-empty bins",
            )

    if args.linear:
        ax.set_yscale("linear")
    else:
        ax.set_yscale("log")

    ax.set_xlabel(f"Interarrival time ({x_unit})")
    ax.set_ylabel("Counts")
    ax.set_title(
        "Interarrival Histogram"
        f"\n{file_path}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    print(f"Loaded: {file_path}")
    print(f"binwidth_ps: {data.get('binwidth_ps')}")
    print(f"display_total_bins: {data.get('display_total_bins')}")
    print(f"special_corr_delay_ps: {data.get('special_corr_delay_ps')}")
    print(f"non-empty raw bins: {len(data.get('raw_sparse_counts', []))}")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
