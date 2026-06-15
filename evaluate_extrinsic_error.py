#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import List, Tuple


# Ground-truth extrinsic (camera <- lidar) for the simulation set.
R_GT = [
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
]
T_GT = [0.0, 0.05, 0.0]  # meters


def transpose_3x3(m: List[List[float]]) -> List[List[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


def matmul_3x3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    out = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(3))
    return out


def parse_line(line: str) -> Tuple[str, List[List[float]], List[float]]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) == 13:
        name = parts[0]
        nums = [float(x) for x in parts[1:]]
    elif len(parts) == 12:
        name = "extrinsic"
        nums = [float(x) for x in parts]
    else:
        raise ValueError(
            f"Invalid column count: expected 12 numeric values or 13 columns "
            f"with a name, got {len(parts)} columns: {line}"
        )

    r = [
        nums[0:3],
        nums[3:6],
        nums[6:9],
    ]
    t = nums[9:12]
    return name, r, t


def rotation_error_deg(r_gt: List[List[float]], r: List[List[float]]) -> float:
    r_delta = matmul_3x3(transpose_3x3(r_gt), r)
    trace = r_delta[0][0] + r_delta[1][1] + r_delta[2][2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def translation_error_cm(t_gt: List[float], t: List[float]) -> float:
    dx = t_gt[0] - t[0]
    dy = t_gt[1] - t[1]
    dz = t_gt[2] - t[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz) * 100.0


def load_single_extrinsic(txt_path: Path) -> Tuple[str, List[List[float]], List[float]]:
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        return parse_line(line)
    raise ValueError("No valid extrinsic line was found in the input file.")


def format_vector(values: List[float]) -> str:
    return "[" + ", ".join(f"{v:.10f}" for v in values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one extrinsic against the ground truth."
    )
    parser.add_argument(
        "txt_path",
        help="Path to a txt file containing one extrinsic. "
        "Expected format: optional name followed by 12 comma-separated values "
        "(3x3 rotation matrix and 3D translation vector).",
    )
    args = parser.parse_args()

    txt_path = Path(args.txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {txt_path}")

    name, r, t = load_single_extrinsic(txt_path)
    et = translation_error_cm(T_GT, t)
    er = rotation_error_deg(R_GT, r)

    print("===== Extrinsic Error =====")
    print(f"Input file: {txt_path}")
    print(f"Extrinsic name: {name}")
    print(f"Estimated translation: {format_vector(t)} m")
    print(f"Ground-truth translation: {format_vector(T_GT)} m")
    print(f"Translation error: {et:.3f} cm")
    print(f"Rotation error: {er:.6f} deg")


if __name__ == "__main__":
    main()
