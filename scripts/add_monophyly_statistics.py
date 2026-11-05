#!/usr/bin/env python3
"""Add rooted query-clade statistics to an existing placement_statistics.csv.

Requires: pip install dendropy
Run from QTMerge (where Datasets/ exists):
 python add_monophyly_statistics.py \
   --input Datasets/large_placement/placement_statistics.csv \
   --output Datasets/large_placement/placement_statistics_monophyly.csv

Each source species tree is rooted using the outgroup from the instance manifest.
All query taxa must be ingroup taxa. No distance matrix is constructed.
"""
import argparse
import csv
import json
from pathlib import Path
import dendropy

NEW_COLUMNS = ["is_monophyletic", "mrca_clade_size", "clade_purity",
               "largest_query_clade_size", "largest_query_clade_fraction"]


def read_rooted_tree(path, outgroup):
    tree = dendropy.Tree.get(path=str(path), schema="newick",
                             rooting="force-unrooted", preserve_underscores=True)
    tips = {leaf.taxon.label: leaf for leaf in tree.leaf_node_iter()}
    if len(tips) != len(list(tree.leaf_node_iter())):
        raise ValueError(f"Duplicate labels in {path}")
    if outgroup not in tips:
        raise ValueError(f"Outgroup {outgroup!r} not found in {path}")
    # Root on the outgroup edge: a clade is defined relative to this root.
    tree.to_outgroup_position(tips[outgroup], update_bipartitions=False)
    return tree, set(tips) - {outgroup}


def clade_stats(tree, ingroup, queries):
    q = len(queries)
    if q < 1 or not queries <= ingroup:
        raise ValueError("Query labels must be a nonempty subset of ingroup labels")
    counts = {}
    largest = 0
    mrca_size = None
    # Counts are (number of ingroup descendants, number of query descendants).
    # For the MRCA, take the deepest node containing all q queries.
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            label = node.taxon.label
            counts[node] = (int(label in ingroup), int(label in queries))
        else:
            child_counts = [counts[c] for c in node.child_node_iter()]
            counts[node] = (sum(x[0] for x in child_counts),
                            sum(x[1] for x in child_counts))
        size, hits = counts[node]
        if size > 0 and size == hits:
            largest = max(largest, size)
        if hits == q and mrca_size is None:
            mrca_size = size
    if mrca_size is None:
        raise RuntimeError("MRCA could not be located")
    return {"is_monophyletic": int(mrca_size == q),
            "mrca_clade_size": mrca_size,
            "clade_purity": q / mrca_size,
            "largest_query_clade_size": largest,
            "largest_query_clade_fraction": largest / q}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("Datasets/large_placement/placement_statistics.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("Datasets/large_placement/placement_statistics_monophyly.csv"))
    parser.add_argument("--root-dir", type=Path, default=Path("."),
                        help="Prefix for paths in CSV, if running outside QTMerge")
    args = parser.parse_args()
    with args.input.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"instance_dir", "source_tree", "q"}
    if not required <= set(fieldnames):
        raise ValueError(f"CSV missing required columns: {required - set(fieldnames)}")
    # Group by source tree to parse/root each original tree once.
    grouped = {}
    for row in rows:
        source = args.root_dir / row["source_tree"]
        grouped.setdefault(source, []).append(row)
    for source, group in grouped.items():
        tree = None
        ingroup = None
        outgroup_used = None
        for row in group:
            instance = args.root_dir / row["instance_dir"]
            manifest = json.loads((instance / "manifest.json").read_text())
            outgroup = str(manifest["outgroup"])
            if tree is None:
                tree, ingroup = read_rooted_tree(source, outgroup)
                outgroup_used = outgroup
            elif outgroup != outgroup_used:
                raise ValueError(f"Conflicting outgroups for {source}")
            queries = [line.strip() for line in (instance / "queries.txt").read_text().splitlines()
                       if line.strip()]
            if len(queries) != int(row["q"]) or len(set(queries)) != len(queries):
                raise ValueError(f"Query count/uniqueness mismatch in {instance}")
            row.update(clade_stats(tree, ingroup, set(queries)))
        print(f"Processed {source}: {len(group)} instances", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames + [x for x in NEW_COLUMNS if x not in fieldnames]
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
