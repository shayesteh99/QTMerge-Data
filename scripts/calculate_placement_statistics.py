#!/usr/bin/env python3
"""Compute MPD, MNTD, SES-MPD and SES-MNTD for generated placement instances.

Dependencies: pip install dendropy numpy
Run from QTMerge:
 python calculate_placement_statistics.py --data-dir Datasets/large \\
     --instances-dir Datasets/large_placement --null-reps 100

SES = (observed - mean(random))/SD(random), using q taxa sampled uniformly
without replacement from the same tree's ingroup. Negative SES means clustering.
Random samples are shared across alpha values for each (tree, q).
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import dendropy
import numpy as np
from tree_distances_numpy import ingroup_distance_matrix


def stable_seed(*items):
    raw = "|".join(map(str, items)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def distances_for_tree(path, outgroup):
    tree = dendropy.Tree.get(path=str(path), schema="newick",
                             rooting="force-unrooted", preserve_underscores=True)
    return ingroup_distance_matrix(tree, outgroup)


def mpd_mntd(matrix, indices):
    sub = matrix[np.ix_(indices, indices)].copy()
    q = len(indices)
    if q < 2:
        return float("nan"), float("nan")
    mpd = float(sub.sum(dtype=np.float64) / (q * (q - 1)))
    np.fill_diagonal(sub, np.inf)
    mntd = float(sub.min(axis=1).mean())
    return mpd, mntd


def null_distribution(matrix, q, repetitions, seed):
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    values = np.empty((repetitions, 2), dtype=np.float64)
    for k in range(repetitions):
        indices = rng.choice(n, size=q, replace=False)
        values[k] = mpd_mntd(matrix, indices)
    return values


def ses(observed, null_values):
    mean = float(np.mean(null_values))
    sd = float(np.std(null_values, ddof=1))
    return mean, sd, ((observed - mean) / sd if sd > 0 else float("nan"))


FIELDS = [
    "n_taxa", "replicate", "q_scheme", "q", "alpha", "seed",
    "query_fraction", "backbone_ingroup_taxa", "mpd", "mntd",
    "null_mpd_mean", "null_mpd_sd", "ses_mpd",
    "null_mntd_mean", "null_mntd_sd", "ses_mntd",
    "null_reps", "instance_dir", "source_tree",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Datasets/large"))
    parser.add_argument("--instances-dir", type=Path, default=Path("Datasets/large_placement"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--null-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 1000, 5000])
    parser.add_argument("--replicates", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    if args.null_reps < 2:
        parser.error("--null-reps must be at least 2")
    output = args.output or (args.instances_dir / "placement_statistics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Use generator's exact directory conventions; do not depend on absolute paths
    # stored in the manifest (which may differ between computers).
    schemes = ("log2", "sqrt", "2sqrt", "n10")
    alphas = (0., .25, .5, .75, 1.)
    rows = []
    for n in args.sizes:
        qs = {"log2": math.ceil(math.log2(n)),
              "sqrt": math.ceil(math.sqrt(n)),
              "2sqrt": math.ceil(2 * math.sqrt(n)),
              "n10": math.ceil(n / 10)}
        for rep in args.replicates:
            rep_name = f"{rep:02d}"
            source = args.data_dir / str(n) / rep_name / "s_tree.trees"
            if not source.is_file():
                if args.skip_missing:
                    print(f"WARNING: missing {source}", file=sys.stderr)
                    continue
                raise FileNotFoundError(source)

            # Discover outgroup from manifest, avoiding a separate CLI argument.
            first_manifest = (args.instances_dir / str(n) / rep_name /
                              f"q_log2_{qs['log2']}" / "alpha_0.00" / "manifest.json")
            if not first_manifest.is_file():
                if args.skip_missing:
                    print(f"WARNING: missing {first_manifest}", file=sys.stderr)
                    continue
                raise FileNotFoundError(first_manifest)
            outgroup = str(json.loads(first_manifest.read_text())["outgroup"])
            ingroup, matrix = distances_for_tree(source, outgroup)
            if len(ingroup) != n:
                raise ValueError(f"{source}: expected {n} ingroup taxa, found {len(ingroup)}")
            lookup = {name: i for i, name in enumerate(ingroup)}

            for scheme in schemes:
                q = qs[scheme]
                null = null_distribution(matrix, q, args.null_reps,
                                         stable_seed(args.seed, n, rep_name, scheme, q))
                for alpha in alphas:
                    folder = (args.instances_dir / str(n) / rep_name /
                              f"q_{scheme}_{q}" / f"alpha_{alpha:.2f}")
                    query_file = folder / "queries.txt"
                    manifest_file = folder / "manifest.json"
                    if not query_file.is_file() or not manifest_file.is_file():
                        if args.skip_missing:
                            print(f"WARNING: incomplete instance {folder}", file=sys.stderr)
                            continue
                        raise FileNotFoundError(f"Incomplete instance: {folder}")
                    manifest = json.loads(manifest_file.read_text())
                    if str(manifest["outgroup"]) != outgroup:
                        raise ValueError(f"Outgroup mismatch: {folder}")
                    queries = [line.strip() for line in query_file.read_text().splitlines()
                               if line.strip()]
                    if len(queries) != q or len(set(queries)) != q:
                        raise ValueError(f"Wrong query count or duplicate labels: {folder}")
                    if set(queries) - set(lookup):
                        raise ValueError(f"Unknown or outgroup query taxon: {folder}")
                    indices = np.array([lookup[x] for x in queries], dtype=np.int64)
                    mpd, mntd = mpd_mntd(matrix, indices)
                    null_mpd_mean, null_mpd_sd, ses_mpd = ses(mpd, null[:, 0])
                    null_mntd_mean, null_mntd_sd, ses_mntd = ses(mntd, null[:, 1])
                    rows.append({
                        "n_taxa": n, "replicate": rep_name, "q_scheme": scheme,
                        "q": q, "alpha": alpha, "seed": manifest.get("seed", ""),
                        "query_fraction": q / n, "backbone_ingroup_taxa": n - q,
                        "mpd": mpd, "mntd": mntd,
                        "null_mpd_mean": null_mpd_mean, "null_mpd_sd": null_mpd_sd,
                        "ses_mpd": ses_mpd,
                        "null_mntd_mean": null_mntd_mean, "null_mntd_sd": null_mntd_sd,
                        "ses_mntd": ses_mntd,
                        "null_reps": args.null_reps,
                        "instance_dir": str(folder), "source_tree": str(source)
                    })
            print(f"Processed n={n} replicate={rep_name}: {len(schemes)*len(alphas)} instances",
                  flush=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
