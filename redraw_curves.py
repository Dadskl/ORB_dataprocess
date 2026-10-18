#!/usr/bin/env python3
"""Redraw diagnostic figures from saved metrics/CSV without recomputing SLAM errors."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from diagnostic_plots import run_plots, batch_plots
from origin_export import Report


def restore_report(path):
    manifest = json.loads((path/'figure_manifest.json').read_text())
    report = Report(path, manifest['formats'], manifest['dpi'], excel=False)
    report.metrics = json.loads((path/'metrics.json').read_text())
    report.images = manifest['figures'].copy()
    for table in (path/'03_data').glob('*.csv'):
        # Only these full-resolution samples are required for diagnostic plots.
        if table.stem not in ('Frames', 'TrackingLosses', 'tracking_time') and not table.stem.endswith(
                ('_absolute_errors', '_relative_errors', '_aligned_speed', '_angular_speed', '_speed', '_acceleration')):
            continue
        with table.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            report.tables[table.stem] = {'headers': next(reader),
                                        'rows': [[None if c == '' else c for c in row] for row in reader]}
    return report


def redraw(report, batch=None):
    previous = report.images.copy()
    report.images = []
    if batch is None: run_plots(report)
    else: batch_plots(report, batch)
    names = report.images.copy()
    report.images = list(dict.fromkeys(previous+names))
    (report.path/'figure_manifest.json').write_text(json.dumps(
        {'figures': report.images, 'dpi': report.dpi, 'formats': report.formats}, indent=2)+'\n')
    print(f'{report.path.name}: redrew {len(names)} diagnostic figures', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path, help='Existing batch output or one completed experiment output')
    args = p.parse_args()
    root = restore_report(args.output)
    if root.metrics.get('sensor_mode'):
        redraw(root)
    else:
        reports = [restore_report(p.parent) for p in sorted(args.output.glob('*/metrics.json'))]
        for report in reports: redraw(report)
        redraw(root, reports)


if __name__ == '__main__': main()
