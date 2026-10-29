"""Print one table with every run in runs/: validation accuracy, train accuracy, best epoch, time and,
where it exists, test accuracy. Seed repeats (<name>_s1, <name>_s2) are also summarized as mean ± std.

Usage:
    python collect_results.py
"""
import json
import re
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(x):
    return f'{100 * x:.1f}' if x is not None else '—'


def main():
    rows = []
    for run in sorted(p for p in Path('runs').iterdir() if p.is_dir()):
        summary = load(run / 'summary.json')
        if summary is None:
            continue
        test = load(run / 'eval_test.json')
        if 'final_knn_acc' in summary:  # SSL pretraining run: report its kNN monitor
            rows.append((run.name, f"kNN {pct(summary['final_knn_acc'])}", '—', '—',
                         f"{summary['train_seconds'] / 60:.0f}", '—'))
        else:
            rows.append((run.name, pct(summary['best_val_acc']), pct(summary.get('final_train_acc')),
                         str(summary['best_epoch']), f"{summary['train_seconds'] / 60:.0f}",
                         pct(test['acc']) if test else '—'))

    header = ('run', 'val %', 'train %', 'best ep', 'min', 'test %')
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    for r in [header] + rows:
        print('  '.join(str(c).ljust(w) for c, w in zip(r, widths)))

    # Seed groups: base run + its _s<k> repeats.
    groups = {}
    for name, val, *_ in rows:
        if val.startswith('kNN'):
            continue
        base = re.sub(r'_s\d+$', '', name)
        groups.setdefault(base, []).append(float(val))
    print('\nSeed repeats (mean ± std of best val accuracy):')
    for base, vals in groups.items():
        if len(vals) > 1:
            print(f'  {base:<24s} {statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f}  '
                  f'(n={len(vals)}: {", ".join(f"{v:.1f}" for v in vals)})')


if __name__ == '__main__':
    main()
