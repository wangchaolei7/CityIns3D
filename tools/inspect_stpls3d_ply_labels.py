import argparse
from collections import Counter

import numpy as np
from plyfile import PlyData


def load_vertex_labels(path):
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    names = vertex.data.dtype.names
    required = {"semantic", "instance", "x", "y", "z"}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"{path} missing required fields: {sorted(missing)}")
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    semantic = np.asarray(vertex["semantic"]).astype(np.int64)
    instance = np.asarray(vertex["instance"]).astype(np.int64)
    return xyz, semantic, instance


def summarize_semantics(name, semantic):
    counts = Counter(map(int, semantic.tolist()))
    print(f"[{name}] semantic labels ({len(counts)} unique):")
    for sem, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  sem={sem:<4d} points={count}")


def summarize_instances(name, semantic, instance, top_k):
    pair_counts = Counter(zip(map(int, semantic.tolist()), map(int, instance.tolist())))
    print(f"[{name}] top {top_k} (semantic, instance) pairs by points:")
    for (sem, inst), count in pair_counts.most_common(top_k):
        print(f"  sem={sem:<4d} inst={inst:<8d} points={count}")


def compare_labels(orig_sem, orig_inst, gt_sem, gt_inst, top_k):
    same_sem = orig_sem == gt_sem
    same_inst = orig_inst == gt_inst
    same_both = same_sem & same_inst

    total = len(orig_sem)
    print("[diff] point-level summary:")
    print(f"  total_points={total}")
    print(f"  same_semantic={int(same_sem.sum())}")
    print(f"  same_instance={int(same_inst.sum())}")
    print(f"  same_both={int(same_both.sum())}")
    print(f"  changed_semantic={int((~same_sem).sum())}")
    print(f"  changed_instance={int((~same_inst).sum())}")
    print(f"  changed_any={int((~same_both).sum())}")

    transition_counts = Counter(
        zip(
            map(int, orig_sem.tolist()),
            map(int, orig_inst.tolist()),
            map(int, gt_sem.tolist()),
            map(int, gt_inst.tolist()),
        )
    )
    changed_transitions = [
        (key, count)
        for key, count in transition_counts.items()
        if not (key[0] == key[2] and key[1] == key[3])
    ]
    changed_transitions.sort(key=lambda item: (-item[1], item[0]))

    print(f"[diff] top {top_k} changed transitions (orig -> gt):")
    for (orig_s, orig_i, gt_s, gt_i), count in changed_transitions[:top_k]:
        print(
            "  "
            f"(sem={orig_s}, inst={orig_i}) -> (sem={gt_s}, inst={gt_i}) "
            f"points={count}"
        )

    semantic_transition_counts = Counter(
        zip(map(int, orig_sem.tolist()), map(int, gt_sem.tolist()))
    )
    semantic_transition_counts = [
        (key, count)
        for key, count in semantic_transition_counts.items()
        if key[0] != key[1]
    ]
    semantic_transition_counts.sort(key=lambda item: (-item[1], item[0]))
    print(f"[diff] top {top_k} semantic-only transitions:")
    for (orig_s, gt_s), count in semantic_transition_counts[:top_k]:
        print(f"  sem {orig_s} -> {gt_s} points={count}")

    orig_to_gt = {}
    gt_to_orig = {}
    for orig_s, orig_i, gt_s, gt_i in zip(
        map(int, orig_sem.tolist()),
        map(int, orig_inst.tolist()),
        map(int, gt_sem.tolist()),
        map(int, gt_inst.tolist()),
    ):
        orig_key = (orig_s, orig_i)
        gt_key = (gt_s, gt_i)
        orig_to_gt.setdefault(orig_key, Counter())[gt_key] += 1
        gt_to_orig.setdefault(gt_key, Counter())[orig_key] += 1

    split_orig = [
        (orig_key, mapping)
        for orig_key, mapping in orig_to_gt.items()
        if len(mapping) > 1
    ]
    merge_gt = [
        (gt_key, mapping)
        for gt_key, mapping in gt_to_orig.items()
        if len(mapping) > 1
    ]

    print("[mapping] original -> groundtruth:")
    print(f"  original instance pairs={len(orig_to_gt)}")
    print(f"  one_to_one={len(orig_to_gt) - len(split_orig)}")
    print(f"  split_to_multiple_gt={len(split_orig)}")
    for orig_key, mapping in sorted(split_orig, key=lambda item: -sum(item[1].values()))[:top_k]:
        targets = ", ".join(
            f"(sem={gt_key[0]}, inst={gt_key[1]}, points={count})"
            for gt_key, count in mapping.most_common(top_k)
        )
        print(f"  original (sem={orig_key[0]}, inst={orig_key[1]}) -> {targets}")

    print("[mapping] groundtruth <- original:")
    print(f"  groundtruth instance pairs={len(gt_to_orig)}")
    print(f"  one_to_one={len(gt_to_orig) - len(merge_gt)}")
    print(f"  merged_from_multiple_original={len(merge_gt)}")
    for gt_key, mapping in sorted(merge_gt, key=lambda item: -sum(item[1].values()))[:top_k]:
        sources = ", ".join(
            f"(sem={orig_key[0]}, inst={orig_key[1]}, points={count})"
            for orig_key, count in mapping.most_common(top_k)
        )
        print(f"  groundtruth (sem={gt_key[0]}, inst={gt_key[1]}) <- {sources}")


def main():
    parser = argparse.ArgumentParser(description="Inspect label differences between two STPLS3D PLY files")
    parser.add_argument("--original", required=True, help="Path to original_ply_files/*.ply")
    parser.add_argument("--groundtruth", required=True, help="Path to groundtruth/*.ply")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top transitions to print")
    parser.add_argument(
        "--check-coords",
        action="store_true",
        help="Check whether xyz coordinates are identical between the two files",
    )
    args = parser.parse_args()

    orig_xyz, orig_sem, orig_inst = load_vertex_labels(args.original)
    gt_xyz, gt_sem, gt_inst = load_vertex_labels(args.groundtruth)

    if len(orig_sem) != len(gt_sem):
        raise ValueError(f"point count mismatch: {len(orig_sem)} vs {len(gt_sem)}")

    if args.check_coords:
        coord_equal = np.allclose(orig_xyz, gt_xyz)
        print(f"[coords] identical={coord_equal}")
        if not coord_equal:
            diff = np.abs(orig_xyz - gt_xyz).max()
            print(f"[coords] max_abs_diff={diff}")

    summarize_semantics("original", orig_sem)
    summarize_semantics("groundtruth", gt_sem)
    summarize_instances("original", orig_sem, orig_inst, args.top_k)
    summarize_instances("groundtruth", gt_sem, gt_inst, args.top_k)
    compare_labels(orig_sem, orig_inst, gt_sem, gt_inst, args.top_k)


if __name__ == "__main__":
    main()
