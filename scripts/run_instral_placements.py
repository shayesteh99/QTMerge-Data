#!/usr/bin/env python3
"""Sequential INSTRAL placements on QTMerge alpha-sampled instances.

Requires Java, an INSTRAL jar, and DendroPy for safe backbone normalization.
Uses the exact query order in queries.txt; updates the backbone after every insertion.
Results live in each instance's instral_<gene-trees>/ directory. Resume is per completed step.
Raw INSTRAL outputs are retained as steps/NNNN.nwk; label-free backbones
are steps/NNNN.backbone.nwk. The latter are used for later insertions.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SCHEMES = ('log2', 'sqrt', '2sqrt', 'n10')
ALPHAS = (0, .25, .5, .75, 1)


def q_sizes(n):
    return {'log2': math.ceil(math.log2(n)), 'sqrt': math.ceil(math.sqrt(n)),
            '2sqrt': math.ceil(2 * math.sqrt(n)), 'n10': math.ceil(n / 10)}


def validate_tree(path, expected):
    import dendropy
    t = dendropy.Tree.get(path=str(path), schema='newick',
                           rooting='force-unrooted', preserve_underscores=True)
    labels = [x.taxon.label for x in t.leaf_node_iter()]
    if len(labels) != len(set(labels)) or set(labels) != expected:
        raise ValueError('Taxon mismatch or duplicate leaf labels in ' + str(path))
    return t


def clean_backbone(raw, clean, expected):
    """Preserve the original INSTRAL output; create a label-free input tree.

    ASTRAL/INSTRAL emits repeated numeric support labels on internal nodes.
    Those labels can collide with each other or leaf names and its own Newick
    reader rejects them on the next iteration. Removing labels does not
    modify the phylogenetic topology or its branch lengths.
    """
    t = validate_tree(raw, expected)
    for node in t.internal_nodes():
        node.label = None
    temp = clean.with_suffix(clean.suffix + '.tmp')
    try:
        t.write(path=str(temp), schema='newick', suppress_rooting=True,
                suppress_internal_node_labels=True)
        reread = validate_tree(temp, expected)
        remaining = [node.label for node in reread.internal_nodes()
                     if node.label is not None]
        if remaining:
            raise ValueError('Internal labels remain in cleaned tree: ' + str(remaining[:10]))
        temp.replace(clean)
    finally:
        temp.unlink(missing_ok=True)


def find_gene_trees(rep_dir, name):
    p = rep_dir / name
    if not p.is_file():
        raise FileNotFoundError(f'Gene-tree input must be a Newick FILE: {p} (not a directory)')
    return p.resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('Datasets/large'))
    p.add_argument('--instances-dir', type=Path, default=Path('Datasets/large_placement'))
    p.add_argument('--jar', type=Path, required=True, help='Path to instral.jar; keep its lib/ alongside it')
    p.add_argument('--java', default='java')
    p.add_argument('--heap-gb', type=int, default=8)
    p.add_argument('--gene-trees', default='estimatedgenetre', choices=['estimatedgenetre', 'truegenetrees'])
    p.add_argument('--sizes', type=int, nargs='+', default=[500, 1000, 5000])
    p.add_argument('--replicates', type=int, nargs='+', default=list(range(1, 21)))
    p.add_argument('--schemes', nargs='+', choices=SCHEMES, default=list(SCHEMES))
    p.add_argument('--alphas', type=float, nargs='+', default=list(ALPHAS))
    p.add_argument('--max-instances', type=int, default=0, help='0 means no limit')
    p.add_argument('--max-queries', type=int, default=0, help='0 means all queries per instance')
    p.add_argument('--validate', action='store_true', help='Deprecated: every tree is now validated')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    jar = args.jar.resolve()
    if not jar.is_file():
        p.error('INSTRAL jar not found: ' + str(jar))
    if shutil.which(args.java) is None:
        p.error('Java executable not found: ' + args.java)
    try:
        import dendropy  # noqa: F401
    except ImportError:
        p.error('DendroPy is required: pip install dendropy')
    if args.heap_gb < 1:
        p.error('--heap-gb must be positive')
    for alpha in args.alphas:
        if alpha not in ALPHAS:
            p.error('Unsupported alpha: ' + str(alpha))

    args.data_dir = args.data_dir.resolve()
    args.instances_dir = args.instances_dir.resolve()
    summary_path = args.instances_dir / ('instral_' + args.gene_trees + '_summary.csv')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = ['n_taxa', 'replicate', 'q_scheme', 'q', 'alpha', 'gene_trees',
                      'completed_queries', 'total_queries', 'status', 'elapsed_seconds',
                      'result_tree', 'error']
    n_instances = 0
    # Append rows as each instance finishes, so interrupted jobs retain progress.
    with summary_path.open('a', newline='') as summary:
        writer = csv.DictWriter(summary, fieldnames=summary_fields)
        if summary_path.stat().st_size == 0:
            writer.writeheader()
        for n in args.sizes:
            for rep in args.replicates:
                rep_str = f'{rep:02d}'
                rep_dir = args.data_dir / str(n) / rep_str
                for scheme in args.schemes:
                    q = q_sizes(n)[scheme]
                    for alpha in args.alphas:
                        if args.max_instances and n_instances >= args.max_instances:
                            return
                        instance = args.instances_dir / str(n) / rep_str / f'q_{scheme}_{q}' / f'alpha_{alpha:.2f}'
                        if not instance.is_dir():
                            print('WARNING: missing instance', instance, file=sys.stderr)
                            continue
                        n_instances += 1
                        row = dict(n_taxa=n, replicate=rep_str, q_scheme=scheme, q=q,
                                   alpha=alpha, gene_trees=args.gene_trees,
                                   completed_queries=0, total_queries=q, status='error',
                                   elapsed_seconds=0, result_tree='', error='')
                        started = time.monotonic()
                        try:
                            queries = [s.strip() for s in (instance / 'queries.txt').read_text().splitlines() if s.strip()]
                            if len(queries) != q or len(set(queries)) != q:
                                raise ValueError('Invalid queries.txt')
                            backbone = (instance / 'backbone.nwk').resolve()
                            if not backbone.is_file():
                                raise FileNotFoundError(backbone)
                            genes = find_gene_trees(rep_dir, args.gene_trees)
                            outdir = (instance / ('instral_' + args.gene_trees)).resolve()
                            steps = (outdir / 'steps').resolve()
                            if not args.dry_run:
                                steps.mkdir(parents=True, exist_ok=True)
                            import dendropy
                            bt = dendropy.Tree.get(path=str(backbone), schema='newick',
                                                   rooting='force-unrooted', preserve_underscores=True)
                            expected = {x.taxon.label for x in bt.leaf_node_iter()}
                            validate_tree(backbone, expected)
                            if any(t in expected for t in queries):
                                raise ValueError('A query taxon is already present in the initial backbone')
                            current = backbone
                            limit = min(q, args.max_queries) if args.max_queries else q
                            for idx, taxon in enumerate(queries[:limit], 1):
                                out = (steps / f'{idx:04d}.nwk').resolve()
                                done = steps / f'{idx:04d}.done.json'
                                clean = (steps / f'{idx:04d}.backbone.nwk').resolve()
                                if taxon in expected:
                                    raise ValueError(f'Query taxon {taxon!r} already in backbone at step {idx}')
                                cmd = [args.java, f'-Xmx{args.heap_gb}g',
                                       f'-Djava.library.path={jar.parent}', '-jar', str(jar),
                                       '-i', str(genes), '-f', str(current),
                                       '--placement', taxon, '-o', str(out)]
                                if args.dry_run:
                                    print(' '.join(cmd))
                                    current = out
                                    continue
                                if args.overwrite:
                                    out.unlink(missing_ok=True)
                                    clean.unlink(missing_ok=True)
                                    done.unlink(missing_ok=True)
                                if out.is_file() and out.stat().st_size > 0 and done.is_file():
                                    metadata = json.loads(done.read_text())
                                    if metadata.get('taxon') == taxon and metadata.get('gene_trees') == str(genes):
                                        next_expected = expected | {taxon}
                                        # Also cleans historical completed steps, so a run
                                        # interrupted at 0002 can resume from raw 0001.
                                        clean_backbone(out, clean, next_expected)
                                        expected = next_expected
                                        current = clean
                                        row['completed_queries'] = idx
                                        continue
                                out.unlink(missing_ok=True)
                                clean.unlink(missing_ok=True)
                                done.unlink(missing_ok=True)
                                t0 = time.monotonic()
                                with (steps / f'{idx:04d}.stdout.log').open('w') as stdout, \
                                     (steps / f'{idx:04d}.stderr.log').open('w') as stderr:
                                    proc = subprocess.run(cmd, cwd=str(jar.parent), stdout=stdout, stderr=stderr)
                                elapsed = time.monotonic() - t0
                                if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
                                    raise RuntimeError(
                                        f'INSTRAL failed at step {idx}, taxon {taxon} '
                                        f'(exit code {proc.returncode}); inspect '
                                        f'{steps / f"{idx:04d}.stdout.log"} and '
                                        f'{steps / f"{idx:04d}.stderr.log"}'
                                    )
                                next_expected = expected | {taxon}
                                clean_backbone(out, clean, next_expected)
                                expected = next_expected
                                done.write_text(json.dumps({'taxon': taxon, 'gene_trees': str(genes),
                                                            'elapsed_seconds': elapsed}) + '\n')
                                current = clean
                                row['completed_queries'] = idx
                                print(f'{n}/{rep_str}/{scheme}/{alpha:.2f} step {idx}/{q}: {elapsed:.1f}s', flush=True)
                            row['status'] = ('dry-run' if args.dry_run else 'complete' if limit == q else 'partial')
                            if not args.dry_run and limit == q:
                                final = outdir / 'instral_species_tree.nwk'
                                shutil.copyfile(current, final)
                                # Keep annotations in a separate archival output.
                                shutil.copyfile(out, outdir / 'instral_species_tree_annotated.nwk')
                                row['result_tree'] = str(final)
                            elif not args.dry_run:
                                row['result_tree'] = str(current)
                        except Exception as exc:
                            row['error'] = str(exc)
                            print('ERROR:', instance, exc, file=sys.stderr)
                        row['elapsed_seconds'] = round(time.monotonic() - started, 3)
                        writer.writerow(row)
                        summary.flush()
                        if row['status'] == 'error':
                            raise RuntimeError(f'Failed instance {instance}: {row["error"]}')
    print('Summary:', summary_path)


if __name__ == '__main__':
    main()
