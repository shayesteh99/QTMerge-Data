#!/usr/bin/env python3
"""Reproducible alpha-based query sampling on a true species tree.

Requires: pip install dendropy numpy

Example:
 python alpha_query_sampling.py --tree s_tree.trees --outgroup OUTGROUP \
     --q 50 --alpha 0.5 --seed 123 --output-prefix sampled
"""
import argparse
import json
import math
from pathlib import Path

import dendropy
import numpy as np


def sample_queries(tree, outgroup, q, alpha, seed):
    """Return selected labels in sampling order and backbone tree.

    Patristic distances use branch lengths in the supplied tree.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    leaves = list(tree.leaf_node_iter())
    labels = [leaf.taxon.label for leaf in leaves]
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate leaf labels")
    if outgroup not in labels:
        raise ValueError(f"Outgroup {outgroup!r} not found")
    ingroup = sorted(label for label in labels if label != outgroup)
    n = len(ingroup)
    if not 1 <= q < n:
        raise ValueError(f"q must satisfy 1 <= q < {n}")
    for edge in tree.preorder_edge_iter():
        if edge.tail_node is not None and (edge.length is None or edge.length < 0):
            raise ValueError("Every non-root edge needs a nonnegative branch length")

    pdm = tree.phylogenetic_distance_matrix()
    taxon_by_label = {leaf.taxon.label: leaf.taxon for leaf in leaves}
    # Store the n x n matrix to permit fast incremental nearest-set distances.
    distances = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = pdm.distance(taxon_by_label[ingroup[i]], taxon_by_label[ingroup[j]])
            distances[i, j] = distances[j, i] = d
    positive = distances[np.triu_indices(n, 1)]
    tau = float(np.median(positive))
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("Median ingroup pairwise distance must be positive and finite")

    rng = np.random.default_rng(seed)
    available = np.ones(n, dtype=bool)
    first = int(rng.integers(n))
    chosen = [first]
    available[first] = False
    nearest = distances[first].copy()

    while len(chosen) < q:
        candidates = np.flatnonzero(available)
        d = nearest[candidates]
        if alpha == 0:
            pick = int(rng.choice(candidates))
        elif alpha == 1:
            minimum = float(d.min())
            ties = candidates[np.isclose(d, minimum, rtol=1e-12, atol=1e-12)]
            pick = int(rng.choice(ties))
        else:
            beta = alpha / (1.0 - alpha)
            log_weights = -beta * d / tau
            weights = np.exp(log_weights - log_weights.max())
            probabilities = weights / weights.sum()
            pick = int(rng.choice(candidates, p=probabilities))
        chosen.append(pick)
        available[pick] = False
        np.minimum(nearest, distances[pick], out=nearest)

    queries = [ingroup[i] for i in chosen]
    backbone = tree.clone(depth=2)
    backbone.prune_taxa_with_labels(queries, update_bipartitions=False)
    backbone.suppress_unifurcations()
    expected = set(labels) - set(queries)
    actual = {leaf.taxon.label for leaf in backbone.leaf_node_iter()}
    if actual != expected:
        raise AssertionError("Backbone taxa do not match expected taxa")
    return queries, backbone, tau


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree', required=True, type=Path)
    parser.add_argument('--outgroup', required=True)
    parser.add_argument('--q', required=True, type=int)
    parser.add_argument('--alpha', required=True, type=float)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    tree = dendropy.Tree.get(path=str(args.tree), schema='newick', rooting='force-unrooted', preserve_underscores=True)
    queries, backbone, tau = sample_queries(tree, args.outgroup, args.q, args.alpha, args.seed)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f'{prefix}.queries.txt').write_text('\n'.join(queries) + '\n')
    Path(f'{prefix}.backbone.nwk').write_text(backbone.as_string(schema='newick', suppress_rooting=True, unquoted_underscores=True))
    manifest = dict(source_tree=str(args.tree), outgroup=args.outgroup, q=args.q,
                    alpha=args.alpha, seed=args.seed, tau=tau, queries=queries,
                    backbone_taxa=len(list(backbone.leaf_node_iter())))
    Path(f'{prefix}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'queries'}, indent=2))


if __name__ == '__main__':
    main()
