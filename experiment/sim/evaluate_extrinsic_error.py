#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from statistics import mean, pstdev
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
    if len(parts) != 13:
        raise ValueError(f"列数错误，期望13列，实际{len(parts)}列: {line}")

    name = parts[0]
    nums = [float(x) for x in parts[1:]]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算外参误差：et=||tgt-t||2, eR=acos((tr(Rgt^T R)-1)/2)。"
    )
    parser.add_argument("txt_path", help="外参txt文件路径（唯一参数）")
    args = parser.parse_args()

    txt_path = Path(args.txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"文件不存在: {txt_path}")

    et_list: List[float] = []
    eR_list: List[float] = []

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        name, r, t = parse_line(line)
        et = translation_error_cm(T_GT, t)
        er = rotation_error_deg(R_GT, r)
        et_list.append(et)
        eR_list.append(er)
        print(f"{name}: et={et:.3f} cm, eR={er:.6f} deg")

    if not et_list:
        raise ValueError("未读取到有效数据行，请检查txt文件内容。")

    et_mean = mean(et_list)
    er_mean = mean(eR_list)
    et_std = pstdev(et_list) if len(et_list) > 1 else 0.0
    er_std = pstdev(eR_list) if len(eR_list) > 1 else 0.0

    print("\n===== 统计结果 =====")
    print(f"平移误差均值: {et_mean:.3f} cm")
    print(f"平移误差标准差: {et_std:.3f} cm")
    print(f"旋转误差均值: {er_mean:.6f} deg")
    print(f"旋转误差标准差: {er_std:.6f} deg")


if __name__ == "__main__":
    main()
