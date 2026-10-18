"""Direct scientific figures from the same numeric samples used by the reports.

Independent maps are drawn as separate traces/distributions. No cross-map error
averaging, smoothing, interpolation, or synthetic zero padding is performed.
"""
import numpy as np
import inspect
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

COLORS = {'mono': '#D97706', 'stereo': '#167D9A', 'rgbd': '#6950A1', 'unknown': '#626B73'}


def boxplot(ax, arrays, labels, **kwargs):
    key = 'tick_labels' if 'tick_labels' in inspect.signature(ax.boxplot).parameters else 'labels'
    return ax.boxplot(arrays, **{key: labels}, **kwargs)


def data(report, key):
    rows = report.tables.get(key, {}).get('rows', [])
    return np.asarray(rows, dtype=float) if rows else np.empty((0, 0))


def maps(report):
    return sorted((k for k, v in report.metrics.items() if k.startswith('map_') and isinstance(v, dict)),
                  key=lambda k: int(k.split('_')[1]))


def setup(ax, title, xlabel='', ylabel=''):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.2, linewidth=.6)
    ax.spines[['top', 'right']].set_visible(False)


def missing(ax, message='No eligible samples'):
    ax.text(.5, .5, message, ha='center', va='center', transform=ax.transAxes, color='#777777')


def trace(ax, x, y, **kwargs):
    """Keep temporal gaps visible, including missing/invalid observations."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x): return
    dt = np.diff(x)
    positive = dt[dt > 0]
    if len(positive):
        gaps = np.flatnonzero(dt > 5*np.median(positive))+1
        x, y = np.insert(x, gaps, np.nan), np.insert(y, gaps, np.nan)
    ax.plot(x, y, linewidth=.75, **kwargs)


def ecdf(ax, values, **kwargs):
    values = np.sort(np.asarray(values, float))
    values = values[np.isfinite(values)]
    if len(values): ax.step(values, np.arange(1, len(values)+1)/len(values), where='post', linewidth=.9, **kwargs)


def finish(report, name, fig, caption):
    fig.suptitle(caption, fontsize=11)
    fig.subplots_adjust(hspace=.45, wspace=.35)
    report.save_chart(name, fig)


def map_legend(fig, stems, colors, bottom=.09):
    handles = [Line2D([], [], color=colors[stem], label=stem, linewidth=1.2) for stem in stems]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .008), ncol=min(8, len(stems)),
               fontsize=5.5, frameon=False, handlelength=1.2, columnspacing=1., labelspacing=.4)
    fig._slam_layout_rect = (0, bottom, 1, 1)


def run_plots(report):
    mode = report.metrics.get('sensor_mode', 'unknown')
    seq = report.metrics.get('sequence') or '?'
    alignment = report.metrics.get('options', {}).get('align', '?')
    title = f'{mode.upper()} | sequence {seq} | {alignment.upper()}'
    frames = report.tables.get('Frames')
    start = 0.
    if frames and frames['rows']:
        f = {key: np.asarray([row[i] for row in frames['rows']], float) for i, key in enumerate(frames['headers'])}
        start = f['timestamp'][0]
        t = f['timestamp']-start
        fig, axes = plt.subplots(3, 2, figsize=(13, 11))
        ax = axes[0, 0]
        for key, label in [('tracking_ms', 'Track'), ('extraction_left_ms', 'Left extraction'), ('extraction_right_ms', 'Right extraction')]:
            trace(ax, t, f[key], label=label)
        setup(ax, 'Tracking and feature extraction', 'Time (s)', 'Duration (ms)'); ax.legend(fontsize=8)
        ax = axes[0, 1]
        for key in ('features_left', 'features_right', 'map_inlier_associations'):
            trace(ax, t, f[key], label=key.replace('_', ' '))
        setup(ax, 'Features and map associations', 'Time (s)', 'Count'); ax.legend(fontsize=8)
        ax = axes[1, 0]
        denominator = f['features_left']+f['features_right']
        ratio = np.divide(f['map_inlier_associations'], denominator, out=np.full(len(t), np.nan), where=denominator > 0)
        trace(ax, t, ratio, color='#167D9A')
        setup(ax, 'Association ratio (diagnostic)', 'Time (s)', 'Associations / extracted features')
        ax = axes[1, 1]
        ax.step(t, f['tracking_state'], where='post', color='#6950A1')
        ax.set_yticks([-1, 0, 1, 2, 3, 4, 5], ['Not ready', 'No image', 'Init', 'OK', 'Recently lost', 'Lost', 'OK KLT'])
        setup(ax, 'Tracking state', 'Time (s)')
        ax = axes[2, 0]
        ax.step(t, f['map_id'], where='post', color='#D97706'); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        setup(ax, 'Active map (online recorder)', 'Time (s)', 'Map ID')
        ax = axes[2, 1]
        if len(t) > 1:
            ax.plot(t[1:], 1000*np.diff(t), label='Input interval', linewidth=.8)
            ax.plot(t[1:], f['tracking_ms'][1:], label='Track duration', linewidth=.6, alpha=.8)
            ax.legend(fontsize=8)
        else: missing(ax)
        setup(ax, 'Timing relative to input rate', 'Time (s)', 'Duration (ms)')
        finish(report, 'performance_timeline', fig, title+' | frame diagnostics')

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        timing = f['tracking_ms'][np.isfinite(f['tracking_ms'])]
        axes[0, 0].hist(timing, bins=min(60, max(5, int(np.sqrt(len(timing))))), color='#167D9A', alpha=.85)
        for quantile, color in [(50, '#D97706'), (95, '#AD2952')]:
            if len(timing): axes[0, 0].axvline(np.percentile(timing, quantile), color=color, linestyle='--', label=f'P{quantile}')
        axes[0, 0].legend(fontsize=8)
        setup(axes[0, 0], 'Track duration histogram', 'Duration (ms)', 'Frames')
        ecdf(axes[0, 1], timing, color='#167D9A')
        setup(axes[0, 1], 'Track duration ECDF', 'Duration (ms)', 'Cumulative fraction')
        states, counts = np.unique(f['tracking_state'], return_counts=True)
        axes[1, 0].bar([str(int(v)) for v in states], counts, color='#6950A1')
        setup(axes[1, 0], 'Recorded tracking states', 'State ID', 'Frames')
        loss = data(report, 'TrackingLosses')
        if len(loss): axes[1, 1].bar(np.arange(1, len(loss)+1), loss[:, 3], color='#AD2952')
        else: missing(axes[1, 1], 'No recorded loss events')
        setup(axes[1, 1], 'Observed tracking losses', 'Loss event', 'Duration (s)')
        finish(report, 'performance_distributions', fig, title+' | duration, states and losses')

    stems = maps(report)
    color = {stem: plt.get_cmap('turbo')((i+.5)/max(len(stems), 1)) for i, stem in enumerate(stems)}
    available = [stem for stem in stems if len(data(report, stem+'_absolute_errors'))]
    if not frames and available: start = min(data(report, stem+'_absolute_errors')[0, 0] for stem in available)
    if not available:
        report.notes.append('无可评价的真值误差样本，跳过误差曲线和分布图。')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        specs = [('absolute_errors', 5, 'ATE', 'Position error (m)'), ('absolute_errors', 6, 'Absolute orientation error', 'Angle (deg)'),
                 ('relative_errors', 2, 'RPE translation', 'Error (m)'), ('relative_errors', 3, 'RPE rotation', 'Angle (deg)')]
        for ax, (suffix, column, label, unit) in zip(axes.flat, specs):
            count = 0
            for stem in available:
                values = data(report, stem+'_'+suffix)
                if len(values):
                    trace(ax, values[:, 0]-start, values[:, column], color=color[stem], label=stem); count += 1
            setup(ax, label, 'Time since first recorded frame (s)', unit)
            if not count: missing(ax)
        if len(available) <= 10: axes[0, 0].legend(fontsize=6, ncol=2)
        else: map_legend(fig, available, color, bottom=.16)
        finish(report, 'error_timeline', fig, title+' | separate traces for independent maps')

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for column, ax in enumerate(axes, 2):
            for stem in available:
                values = data(report, stem+'_absolute_errors')
                trace(ax, values[:, 0]-start, values[:, column], color=color[stem])
            ax.axhline(0, color='#666666', linewidth=.5)
            setup(ax, 'XYZ'[column-2]+' position residual', ylabel='Residual (m)')
        axes[-1].set_xlabel('Time since first recorded frame (s)')
        map_legend(fig, available, color, bottom=.14 if len(available) > 16 else .09)
        finish(report, 'position_residuals', fig, title+' | aligned estimate minus ground truth')

        # Paginate map distributions so labels remain readable for fragmented runs.
        for page, offset in enumerate(range(0, len(available), 16), 1):
            group = available[offset:offset+16]
            fig, axes = plt.subplots(3, 2, figsize=(13, 12))
            for row, (suffix, column, label, unit) in enumerate([specs[0], specs[2], specs[3]]):
                selected = [(stem, data(report, stem+'_'+suffix)[:, column]) for stem in group if len(data(report, stem+'_'+suffix))]
                if selected:
                    boxes = boxplot(axes[row, 0], [x[1] for x in selected], [x[0].replace('map_', '') for x in selected], patch_artist=True, showfliers=True,
                                    flierprops={'marker': '.', 'markersize': 2, 'alpha': .45})
                    for box, (stem, values) in zip(boxes['boxes'], selected):
                        box.set_facecolor(color[stem]); box.set_alpha(.65)
                        ecdf(axes[row, 1], values, color=color[stem], label=stem)
                    if len(selected) > 8: axes[row, 0].tick_params(axis='x', labelrotation=45)
                else:
                    missing(axes[row, 0]); missing(axes[row, 1])
                setup(axes[row, 0], label+' by map', 'Independent map ID', unit)
                setup(axes[row, 1], label+' ECDF', unit, 'Cumulative fraction')
            map_legend(fig, group, color, bottom=.08)
            finish(report, f'error_distributions_{page:02d}', fig, title+' | per-map samples; box whiskers = 1.5 IQR')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, suffix, label, unit in zip(axes.flat, ['aligned_speed', 'angular_speed', 'speed', 'acceleration'],
                                     ['Aligned speed', 'Angular speed', 'Raw trajectory speed', 'Raw acceleration'],
                                     ['m/s', 'deg/s', 'map units/s', 'map units/s²']):
        count = 0
        for stem in stems:
            values = data(report, stem+'_'+suffix)
            if len(values):
                offset = report.metrics[stem].get('start_timestamp', start)-start
                trace(ax, values[:, 0]+offset, values[:, 1], color=color[stem]); count += 1
        setup(ax, label, 'Time since first recorded frame (s)', unit)
        if not count: missing(ax)
    if stems: finish(report, 'motion_diagnostics', fig, title+' | final optimized trajectory differences')
    else: plt.close(fig)
    report.notes.append('误差图按独立地图分别绘制，时间轴相对实验首个输入时间戳；时间间隔超过中位间隔 5 倍时断线，不平滑或插值。分布图分页显示地图，箱线图须为 1.5 IQR，不代表置信区间。')


def nested(d, key):
    for part in key.split('.'):
        if not isinstance(d, dict): return None
        d = d.get(part)
    return d


def batch_plots(book, reports):
    reports = [r for r in reports if r.metrics.get('sequence') is not None]
    if not reports: return
    modes = sorted({r.metrics.get('sensor_mode', 'unknown') for r in reports})
    offsets = dict(zip(modes, np.linspace(-.24, .24, len(modes)) if len(modes) > 1 else [0]))
    handles = [Line2D([], [], marker='o', color=COLORS.get(mode, '#626B73'), linestyle='', label=mode.upper()) for mode in modes]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [('frames.tracking_ms.mean', 'Mean Track duration', 'ms'), ('frames.tracking_ms.p95', 'P95 Track duration', 'ms'),
             ('frames.valid_pose_ratio', 'Valid pose ratio', 'Ratio'), ('tracking_losses.count', 'Tracking loss events', 'Count')]
    for ax, (key, label, unit) in zip(axes.flat, specs):
        for report in reports:
            mode = report.metrics.get('sensor_mode', 'unknown'); value = nested(report.metrics, key)
            if value is not None: ax.scatter(int(report.metrics['sequence'])+offsets[mode], value, color=COLORS.get(mode, '#626B73'), s=28)
        setup(ax, label, 'KITTI sequence', unit); ax.set_xticks(range(11), [f'{i:02d}' for i in range(11)])
    axes[0, 0].legend(handles=handles, fontsize=8)
    finish(book, 'comparison_performance', fig, 'Experiment performance | one point per run')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [('accuracy.ate_m.rmse', 'ATE RMSE', 'm'), ('accuracy.rpe.translation_m.rmse', 'RPE translation RMSE', 'm'),
             ('accuracy.rpe.rotation_deg.rmse', 'RPE rotation RMSE', 'deg'), ('accuracy.gt_coverage', 'Ground-truth coverage per map', 'Ratio')]
    for ax, (key, label, unit) in zip(axes.flat, specs):
        for report in reports:
            mode = report.metrics.get('sensor_mode', 'unknown')
            values = [nested(report.metrics[stem], key) for stem in maps(report)]
            values = [v for v in values if v is not None]
            jitter = np.linspace(-.06, .06, len(values)) if len(values) > 1 else [0]
            if values: ax.scatter(int(report.metrics['sequence'])+offsets[mode]+np.asarray(jitter), values, color=COLORS.get(mode, '#626B73'), s=12, alpha=.65)
        setup(ax, label, 'KITTI sequence', unit); ax.set_xticks(range(11), [f'{i:02d}' for i in range(11)])
    axes[0, 0].legend(handles=handles, fontsize=8)
    alignments = ', '.join(f'{mode}: '+ '/'.join(sorted({r.metrics.get('options', {}).get('align', '?').upper() for r in reports if r.metrics.get('sensor_mode', 'unknown') == mode})) for mode in modes)
    finish(book, 'comparison_accuracy', fig, 'Each point = one independent map | '+alignments)

    # Distribution of timing is per experiment, not repeated for each exported map.
    fig, axes = plt.subplots(len(modes), 1, figsize=(13, 3.4*len(modes)), squeeze=False)
    for ax, mode in zip(axes.flat, modes):
        selected = sorted([r for r in reports if r.metrics.get('sensor_mode', 'unknown') == mode], key=lambda r: r.metrics['sequence'])
        selected = [(r, data(r, 'tracking_time')) for r in selected if len(data(r, 'tracking_time'))]
        if selected:
            arrays = [a[:, 1] for r, a in selected]
            # Violin KDE is undefined for constant / singleton data; always keep box plots valid.
            nonconstant = [(i+1, a) for i, a in enumerate(arrays) if len(a) > 1 and np.ptp(a) > 0]
            if nonconstant:
                violin = ax.violinplot([a for i, a in nonconstant], positions=[i for i, a in nonconstant], showextrema=False)
                for body in violin['bodies']: body.set_facecolor(COLORS.get(mode, '#626B73')); body.set_alpha(.3)
            boxplot(ax, arrays, [r.metrics['sequence'] for r, a in selected], widths=.15, showfliers=True,
                    flierprops={'marker': '.', 'markersize': 2, 'alpha': .45})
        else: missing(ax)
        setup(ax, mode.upper()+' Track duration distribution', 'Sequence (each box is one run)', 'ms')
    finish(book, 'comparison_timing_distributions', fig, 'Per-frame timing | violin density and box plots')
    book.notes.append('跨序列误差图每个点代表一个独立对齐地图，不对多地图误差取平均；性能图每个点代表一次实验。单目 Sim3 与双目/RGB-D SE3 的尺度口径不同，须结合覆盖率阅读。')
