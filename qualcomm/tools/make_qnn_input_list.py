"""Build the input_list.txt expected by qairt-quantizer / qnn-net-run.

Pairs the .raw files written by ``export_qnn_raw_inputs.py`` so that each
calibration entry binds together exactly one ``rgb`` tensor and one
``robot_state`` tensor.  Matches the layout used by Listing 2 in the report.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List


_INDEX_RE = re.compile(r"_(\d+)\.raw$")


def _index_for(path: Path) -> int:
    m = _INDEX_RE.search(path.name)
    if not m:
        raise ValueError(f"unexpected calibration filename {path.name}")
    return int(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input names matching the ONNX graph (e.g. rgb robot_state).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    grouped: Dict[str, Dict[int, Path]] = {n: {} for n in args.inputs}
    for path in raw_dir.iterdir():
        for name in args.inputs:
            if path.name.startswith(name + "_"):
                grouped[name][_index_for(path)] = path
                break

    common_indices = sorted(set.intersection(*(set(g.keys()) for g in grouped.values())))
    if not common_indices:
        raise SystemExit("No overlapping calibration indices across inputs.")

    lines: List[str] = []
    for idx in common_indices:
        parts = [f"{name}:={grouped[name][idx]}" for name in args.inputs]
        lines.append(" ".join(parts))

    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"[make_qnn_input_list] wrote {len(lines)} entries to {args.output}")


if __name__ == "__main__":
    main()
