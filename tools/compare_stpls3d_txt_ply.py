#!/usr/bin/env python3
import argparse
import csv
import os
from collections import Counter

import numpy as np
from plyfile import PlyData


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare point count / semantic count / instance count between STPLS3D txt and ply files."
    )
    parser.add_argument(
        "--txt",
        required=True,
        help="Path to txt file, e.g. Synthetic_v3_InstanceSegmentation/*.txt",
    )
    parser.add_argument(
        "--ply",
        required=True,
        help="Path to ply file, e.g. Synthetic_v3_InstanceSegmentation/*.ply",
    )
    parser.add_argument(
        "--txt-sem-col",
        type=int,
        default=6,
        help="0-based semantic column index in txt.",
    )
    parser.add_argument(
        "--txt-inst-col",
        type=int,
        default=7,
        help="0-based instance column index in txt.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="How many class counts to print.",
    )
    return parser.parse_args()


def count_txt(path, sem_col=6, inst_col=7):
    sem_counter = Counter()
    inst_counter = Counter()
    n_rows = 0
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            n_rows += 1
            sem = int(float(row[sem_col]))
            inst = int(float(row[inst_col]))
            sem_counter[sem] += 1
            inst_counter[inst] += 1
    return n_rows, sem_counter, inst_counter


def count_ply(path):
    ply = PlyData.read(path)["vertex"]
    names = ply.data.dtype.names
    sem_name = "semantic" if "semantic" in names else "class"
    inst_name = "instance"
    sem = np.asarray(ply[sem_name]).astype(np.int64)
    inst = np.asarray(ply[inst_name]).astype(np.int64)
    sem_counter = Counter(sem.tolist())
    inst_counter = Counter(inst.tolist())
    return len(sem), sem_counter, inst_counter, sem_name


def print_counter(title, counter, top_k=30):
    print(title)
    for key, value in sorted(counter.items())[:top_k]:
        print(f"  {key}: {value}")


def main():
    args = parse_args()
    txt_rows, txt_sem, txt_inst = count_txt(
        args.txt,
        sem_col=args.txt_sem_col,
        inst_col=args.txt_inst_col,
    )
    ply_rows, ply_sem, ply_inst, ply_sem_name = count_ply(args.ply)

    print(f"[txt] path={args.txt}")
    print(f"  rows={txt_rows}")
    print(f"  unique_semantic={len(txt_sem)}")
    print(f"  unique_instance={len(txt_inst)}")
    print(f"  ignore_instance_count={txt_inst.get(-100, 0)}")
    print_counter("  semantic_counts", txt_sem, top_k=args.top_k)

    print(f"[ply] path={args.ply}")
    print(f"  rows={ply_rows}")
    print(f"  semantic_field={ply_sem_name}")
    print(f"  unique_semantic={len(ply_sem)}")
    print(f"  unique_instance={len(ply_inst)}")
    print_counter("  semantic_counts", ply_sem, top_k=args.top_k)

    print("[compare]")
    print(f"  row_ratio_ply_over_txt={ply_rows / max(txt_rows, 1):.4f}")
    print(f"  txt_only_semantics={sorted(set(txt_sem) - set(ply_sem))}")
    print(f"  ply_only_semantics={sorted(set(ply_sem) - set(txt_sem))}")
    shared = sorted(set(txt_sem) & set(ply_sem))
    print(f"  shared_semantics={shared}")


if __name__ == "__main__":
    main()
