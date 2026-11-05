#!/usr/bin/env python3
"""Generate clade-aware QT-Merge query sets and pruned backbones.

Requires: pip install dendropy numpy
For alpha in (0,1), alpha is TARGET MRCA clade purity, not a guarantee.
For alpha=1, require a single exact-size clade or mark unavailable.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import dendropy

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SCHEMES = ('log2', 'sqrt', '2sqrt', 'n10')

def q_sizes(n):
    return dict(log2=math.ceil(math.log2(n)), sqrt=math.ceil(math.sqrt(n)),
                **{'2sqrt': math.ceil(2 * math.sqrt(n)), 'n10': math.ceil(n / 10)})

def seed_for(base, n, rep, scheme, alpha):
    b = f'{base}|{n}|{rep}|{scheme}|{alpha:.2f}'.encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], 'big')

def root_on_outgroup(tree, outgroup):
    leaves = [x for x in tree.leaf_node_iter() if x.taxon.label == outgroup]
    if len(leaves) != 1:
        raise ValueError(f'Expected one outgroup {outgroup!r}, got {len(leaves)}')
    tree.to_outgroup_position(leaves[0], update_bipartitions=False)

def collect_clades(tree, outgroup):
    """Return ingroup-only descendant clades in a tree rooted at outgroup."""
    desc = {}
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            desc[node] = frozenset([node.taxon.label])
        else:
            desc[node] = frozenset().union(*(desc[ch] for ch in node.child_node_iter()))
    ingroup = desc[tree.seed_node] - {outgroup}
    clades = []
    for node, taxa in desc.items():
        if taxa and outgroup not in taxa and taxa <= ingroup:
            clades.append((node, taxa))
    return ingroup, clades, desc

def mrca_purity(tree, query, outgroup):
    mrca = tree.mrca(taxon_labels=list(query))
    leaves = {x.taxon.label for x in mrca.leaf_iter()} - {outgroup}
    return len(query) / len(leaves), len(leaves)

def choose_queries(tree, ingroup, clades, q, alpha, rng):
    all_taxa = sorted(ingroup)
    if alpha == 0:
        return rng.sample(all_taxa, q), 'random', None
    if alpha == 1:
        exact = [taxa for _, taxa in clades if len(taxa) == q]
        if exact:
            selected = rng.choice(exact)
            return rng.sample(sorted(selected), q), 'single_monophyletic_clade', q
        return None, 'unavailable_no_exact_clade', None
    # Choose a clade of at least q tips whose size approximates q/alpha.
    # Favor closer target purity; break ties randomly.
    eligible = [(node, taxa) for node, taxa in clades if len(taxa) >= q]
    target_size = q / alpha
    if not eligible:
        raise ValueError('No candidate clade contains q taxa')
    best_error = min(abs(len(taxa) - target_size) for _, taxa in eligible)
    candidates = [(node, taxa) for node, taxa in eligible
                  if abs(len(taxa) - target_size) == best_error]
    node, taxa = rng.choice(candidates)
    # Force query MRCA to be this clade's root: include one tip from each
    # of two child branches, then sample the remaining q-2 uniformly.
    children = [sorted(x.taxon.label for x in ch.leaf_iter())
                for ch in node.child_node_iter()]
    if q >= 2 and len(children) >= 2:
        chosen_children = rng.sample(children, 2)
        anchors = [rng.choice(x) for x in chosen_children]
        rest = rng.sample(sorted(set(taxa) - set(anchors)), q - 2)
        chosen = anchors + rest
    else:
        chosen = rng.sample(sorted(taxa), q)
    rng.shuffle(chosen)
    return chosen, 'target_mrca_purity', len(taxa)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=Path('Datasets/large'))
    ap.add_argument('--output-dir', type=Path, default=Path('Datasets/large_placement_clade'))
    ap.add_argument('--outgroup', default='0')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sizes', type=int, nargs='+', default=[500, 1000, 5000])
    ap.add_argument('--replicates', type=int, nargs='+', default=list(range(1, 21)))
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.sizes:
        for rep in args.replicates:
            src = args.data_dir / str(n) / f'{rep:02d}' / 's_tree.trees'
            tree = dendropy.Tree.get(path=str(src), schema='newick',
                                     rooting='force-unrooted', preserve_underscores=True)
            root_on_outgroup(tree, args.outgroup)
            ingroup, clades, _ = collect_clades(tree, args.outgroup)
            if len(ingroup) != n:
                raise ValueError(f'{src}: expected {n} ingroup taxa, got {len(ingroup)}')
            for scheme, q in q_sizes(n).items():
                for alpha in ALPHAS:
                    folder = args.output_dir / str(n) / f'{rep:02d}' / f'q_{scheme}_{q}' / f'alpha_{alpha:.2f}'
                    paths = [folder / 'queries.txt', folder / 'backbone.nwk', folder / 'manifest.json']
                    if paths[2].exists() and not args.overwrite and (all(p.exists() for p in paths) or json.loads(paths[2].read_text()).get('status') == 'unavailable'):
                        manifest = json.loads(paths[2].read_text())
                    else:
                        if any(p.exists() for p in paths) and not args.overwrite:
                            raise FileExistsError(f'Partial instance: {folder}; use --overwrite')
                        seed = seed_for(args.seed, n, rep, scheme, alpha)
                        rng = random.Random(seed)
                        queries, mode, target_clade = choose_queries(tree, ingroup, clades, q, alpha, rng)
                        if queries is None:
                            folder.mkdir(parents=True, exist_ok=True)
                            for stale in paths[:2]:
                                stale.unlink(missing_ok=True)
                            manifest = dict(n_ingroup=n, replicate=f'{rep:02d}', q_scheme=scheme,
                                            q=q, alpha=alpha, seed=seed, source_tree=str(src),
                                            outgroup=args.outgroup, query_taxa=[], backbone_taxa=None,
                                            sampling_mode=mode, target_clade=None,
                                            realized_clade_purity=None, mrca_clade_size=None,
                                            is_monophyletic=False, status='unavailable')
                            paths[2].write_text(json.dumps(manifest, indent=2) + '\n')
                        else:
                            query_set = set(queries)
                            if len(query_set) != q or args.outgroup in query_set:
                                raise AssertionError('Invalid query set')
                            purity, mrca_size = mrca_purity(tree, query_set, args.outgroup)
                            monophyletic = mrca_size == q
                            if alpha == 1 and not monophyletic:
                                raise AssertionError('Monophyly check failed')
                            backbone = tree.clone(depth=2)
                            backbone.prune_taxa_with_labels(queries, update_bipartitions=False)
                            backbone.suppress_unifurcations()
                            remaining = {x.taxon.label for x in backbone.leaf_node_iter()}
                            if remaining != (ingroup - query_set) | {args.outgroup}:
                                raise AssertionError('Backbone mismatch')
                            folder.mkdir(parents=True, exist_ok=True)
                            paths[0].write_text('\n'.join(queries) + '\n')
                            paths[1].write_text(backbone.as_string(schema='newick', suppress_rooting=True,
                                                                 unquoted_underscores=True))
                            manifest = dict(n_ingroup=n, replicate=f'{rep:02d}', q_scheme=scheme,
                                            q=q, alpha=alpha, seed=seed, source_tree=str(src),
                                            outgroup=args.outgroup, query_taxa=queries,
                                            backbone_taxa=len(remaining), sampling_mode=mode,
                                            target_clade=target_clade, realized_clade_purity=purity,
                                            mrca_clade_size=mrca_size, is_monophyletic=monophyletic, status="ready")
                            paths[2].write_text(json.dumps(manifest, indent=2) + '\n')
                    rows.append({k: manifest.get(k, 'ready' if k == 'status' else None) for k in ('n_ingroup', 'replicate', 'q_scheme', 'q',
                                                         'alpha', 'sampling_mode', 'realized_clade_purity',
                                                         'mrca_clade_size', 'is_monophyletic', 'status')}
                                | {'instance_dir': str(folder)})
            print(f'Completed n={n} rep={rep:02d}: 20 instances', flush=True)
    out = args.output_dir / 'instances.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {len(rows)} instances to {out}')

if __name__ == '__main__':
    main()
