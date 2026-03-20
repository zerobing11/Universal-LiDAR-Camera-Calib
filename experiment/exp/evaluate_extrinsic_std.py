#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from statistics import pstdev
from typing import List, Tuple


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
    t = nums[9:12]  # meters
    return name, r, t


def rotation_angle_deg(r: List[List[float]]) -> float:
    """
    把旋转矩阵转换为轴角中的角度（单位：deg）。
    用这个标量角度统计旋转分量的标准差。
    """
    trace = r[0][0] + r[1][1] + r[2][2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估外参分量标准差：旋转(1个, deg) 与平移xyz(3个, cm)。"
    )
    parser.add_argument("txt_path", help="外参txt文件路径（唯一参数）")
    args = parser.parse_args()

    txt_path = Path(args.txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"文件不存在: {txt_path}")

    rot_deg_list: List[float] = []
    tx_cm_list: List[float] = []
    ty_cm_list: List[float] = []
    tz_cm_list: List[float] = []

    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        _, r, t = parse_line(line)
        rot_deg_list.append(rotation_angle_deg(r))
        tx_cm_list.append(t[0] * 100.0)
        ty_cm_list.append(t[1] * 100.0)
        tz_cm_list.append(t[2] * 100.0)

    if not rot_deg_list:
        raise ValueError("未读取到有效数据行，请检查txt文件内容。")

    rot_std_deg = pstdev(rot_deg_list) if len(rot_deg_list) > 1 else 0.0
    tx_std_cm = pstdev(tx_cm_list) if len(tx_cm_list) > 1 else 0.0
    ty_std_cm = pstdev(ty_cm_list) if len(ty_cm_list) > 1 else 0.0
    tz_std_cm = pstdev(tz_cm_list) if len(tz_cm_list) > 1 else 0.0

    print("===== 外参分量标准差 =====")
    print(f"旋转标准差: {rot_std_deg:.6f} deg")
    print(f"平移X标准差: {tx_std_cm:.6f} cm")
    print(f"平移Y标准差: {ty_std_cm:.6f} cm")
    print(f"平移Z标准差: {tz_std_cm:.6f} cm")


if __name__ == "__main__":
    main()
