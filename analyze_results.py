#!/usr/bin/env python3
"""Offline mono / stereo / RGB-D SLAM experiment analysis; see README.md."""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import tempfile
from pathlib import Path
from input_sources import collect_inputs

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostic_plots import run_plots
from matplotlib.ticker import MaxNLocator
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'count': 0}
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x)),
                rmse=float(np.sqrt(np.mean(x*x))), std=float(x.std()),
                min=float(x.min()), max=float(x.max()),
                p90=float(np.percentile(x, 90)), p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)))


def table(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(row[k]) for row in rows]) for k in rows[0]} if rows else {}


def numeric(path, columns=None):
    a = np.loadtxt(path, ndmin=2)
    if not a.size or not np.isfinite(a).all() or (columns and a.shape[1] != columns):
        raise ValueError(f'Invalid or empty numeric data: {path}')
    return a


def trajectory(path, times=None):
    a = numeric(path)
    if a.shape[1] == 8:
        t, p = a[:, 0], a[:, 1:4]
        q = a[:, 4:8]
        if np.any(np.linalg.norm(q, axis=1) < 1e-10):
            raise ValueError(f'Zero quaternion: {path}')
        r = Rotation.from_quat(q).as_matrix()
    elif a.shape[1] == 12:
        if times is None:
            raise ValueError('KITTI ground truth requires --gt-times; row indices are not timestamps.')
        t = numeric(times, 1).ravel()
        if len(t) != len(a):
            raise ValueError('KITTI pose and timestamp counts differ.')
        pose = a.reshape(-1, 3, 4)
        p, r = pose[:, :, 3], pose[:, :, :3]
        if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-3) or np.any(np.linalg.det(r) < 0):
            raise ValueError('Invalid KITTI rotation matrices.')
    else:
        raise ValueError(f'Expected TUM (8) or KITTI (12) columns: {path}')
    if np.any(np.diff(t) <= 0):
        raise ValueError(f'Timestamps must be strictly increasing: {path}')
    return t, p, r


def associate(t, reference, tolerance):
    """Nearest timestamp candidates, smallest residual first, unique matches."""
    right = np.searchsorted(reference, t)
    candidates = []
    for i, j in enumerate(right):
        for k in (j-1, j):
            if 0 <= k < len(reference) and abs(t[i]-reference[k]) <= tolerance:
                candidates.append((abs(t[i]-reference[k]), i, k))
    used_i, used_j, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_i and j not in used_j:
            pairs.append((i, j)); used_i.add(i); used_j.add(j)
    return np.array(sorted(pairs), dtype=int).reshape(-1, 2)


def align(p, g, mode):
    """Least squares proper rotation: aligned = scale * p @ R.T + translation."""
    if mode == 'none':
        return 1., np.eye(3), np.zeros(3)
    x, y = p-p.mean(0), g-g.mean(0)
    if len(p) < 3 or np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError('Alignment needs at least three non-collinear matched positions; use --align none if already registered.')
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    d = np.ones(3); d[-1] = np.linalg.det(u @ vt)
    r = u @ np.diag(d) @ vt
    scale = float((singular*d).sum() / np.mean(np.sum(x*x, axis=1))) if mode == 'sim3' else 1.
    return scale, r, g.mean(0)-scale*(r @ p.mean(0))


def relative_error(p, r, g, gr, i, j):
    re = r[i].T @ r[j]
    rg = gr[i].T @ gr[j]
    te = r[i].T @ (p[j]-p[i])
    tg = gr[i].T @ (g[j]-g[i])
    return np.linalg.norm(te-tg), np.degrees(Rotation.from_matrix(rg.T @ re).magnitude())


def cloud(path):
    """Read XYZ text or ASCII PLY with named vertex properties (no silent binary parsing)."""
    if path.suffix.lower() != '.ply':
        return numeric(path)[:, :3]
    with path.open() as f:
        if f.readline().strip() != 'ply' or f.readline().strip() != 'format ascii 1.0':
            raise ValueError(f'Only ASCII PLY is supported: {path}')
        names, count, vertex = [], None, False
        for line in f:
            s = line.split()
            if s[:2] == ['element', 'vertex']:
                count, vertex = int(s[2]), True
            elif s[:1] == ['element']:
                vertex = False
            elif s[:1] == ['property'] and vertex:
                if s[1] == 'list':
                    raise ValueError('List vertex properties are unsupported.')
                names.append(s[-1])
            elif s == ['end_header']:
                break
        if count is None or not all(k in names for k in ('x', 'y', 'z')):
            raise ValueError(f'Missing PLY vertices: {path}')
        if count == 0:
            return np.empty((0, 3))
        a = np.loadtxt(f, max_rows=count, ndmin=2)
        if len(a) != count or a.shape[1] != len(names):
            raise ValueError(f'PLY vertex count/properties mismatch: {path}')
        a = a[:, [names.index(k) for k in ('x', 'y', 'z')]]
        if not np.isfinite(a).all():
            raise ValueError(f'Non-finite cloud points: {path}')
        return a


from origin_export import Report, comparison, runtime


def plot_trajectory(report, name, p, g=None, color=None):
    report.add(name+'_xyz', ['est_x_map_units', 'est_y_map_units', 'est_z_map_units'] +
               (['gt_x_m', 'gt_y_m', 'gt_z_m'] if g is not None else []) +
               (['ate_m'] if color is not None else []),
               zip(*p.T, *(g.T if g is not None else []), *([color] if color is not None else [])),
               'XY / XZ / YZ trajectory', 'est_x_map_units', 'est_z_map_units',
               'Aligned trajectory coordinates are meters when GT is supplied.')
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(221, projection='3d')
    ax.plot(*p.T, lw=.8, label='Estimated')
    if g is not None: ax.plot(*g.T, lw=.8, label='Ground truth')
    ax.set(xlabel='X', ylabel='Y', zlabel='Z'); ax.legend()
    extent = np.ptp(p if g is None else np.vstack((p, g)), axis=0)
    ax.set_box_aspect(np.maximum(extent, max(extent.max(), 1)*.3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    for k, (i, j) in enumerate(((0, 2), (0, 1), (1, 2)), 2):
        ax = fig.add_subplot(2, 2, k)
        ax.plot(p[:, i], p[:, j], lw=.8, label='Estimated')
        if g is not None: ax.plot(g[:, i], g[:, j], lw=.8, label='Ground truth')
        if color is not None:
            im = ax.scatter(p[:, i], p[:, j], c=color, s=4, cmap='turbo')
            fig.colorbar(im, ax=ax, label='Position error (m)')
        ax.scatter(p[0, i], p[0, j], marker='o', c='green', label='Start')
        ax.scatter(p[-1, i], p[-1, j], marker='x', c='red', label='End')
        ax.set(xlabel='XYZ'[i], ylabel='XYZ'[j]); ax.axis('equal'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    report.save(name, fig)


def analyze_frames(root, report):
    path = root / 'statistics/frames.csv'
    if not path.exists():
        report.notes.append('缺少 frames.csv，跳过逐帧性能分析。'); return
    f = table(path)
    if not f:
        report.notes.append('frames.csv 为空。'); return
    report.add('Frames', list(f), zip(*f.values()), 'Scatter / line', 'timestamp', 'tracking_ms; features_left; map_inlier_associations')
    t = f['timestamp']-f['timestamp'][0]
    report.metrics['frames'] = dict(count=len(t), valid_pose_ratio=float(f['valid_pose'].mean()),
        tracking_ms=stats(f['tracking_ms']), state_counts={str(int(k)): int(v) for k, v in zip(*np.unique(f['tracking_state'], return_counts=True))})
    report.lines('tracking_time', t, {k: f[k] for k in ('tracking_ms', 'extraction_left_ms', 'extraction_right_ms')}, 'Duration (ms)')
    report.distribution('tracking_time_distribution', f['tracking_ms'], 'Track call duration (ms)')
    report.lines('features_and_associations', t, {k: f[k] for k in ('features_left', 'features_right', 'map_inlier_associations')}, 'Count')
    denominator = f['features_left']+f['features_right']
    ratio = np.divide(f['map_inlier_associations'], denominator, out=np.full(len(t), np.nan), where=denominator>0)
    report.lines('association_ratio', t, {'map associations / extracted features': ratio}, 'Ratio (diagnostic only)')
    report.distribution('association_distribution', f['map_inlier_associations'], 'Map-point associations')
    report.lines('tracking_states', t, {'tracking state': f['tracking_state'], 'valid pose': f['valid_pose']}, 'State / boolean')
    report.lines('active_map', t, {'online map ID': f['map_id']}, 'Map ID')
    good = f['tracking_ms'] > 0
    report.lines('inverse_tracking_time', t[good], {'1000 / tracking_ms': 1000/f['tracking_ms'][good]}, 'Call capacity (Hz), NOT end-to-end FPS')
    if len(t) > 1:
        dt = np.diff(t)
        report.lines('input_intervals', t[1:], {'input timestamp interval': dt}, 'Interval (s)')
        mask = dt > 0
        report.metrics['frames']['nonpositive_timestamp_intervals'] = int((~mask).sum())
        report.metrics['frames']['track_exceeds_input_interval_ratio'] = float(np.mean(f['tracking_ms'][1:][mask] > 1000*dt[mask])) if mask.any() else None
        report.metrics['frames']['input_interval_s'] = stats(dt[mask])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ('features_left', 'map_inlier_associations')):
        ax.scatter(f[key], f['tracking_ms'], s=3, alpha=.3)
        ax.set(xlabel=key, ylabel='Track call (ms)'); ax.grid(alpha=.25)
    report.save('timing_vs_features', fig)
    losses = root / 'statistics/tracking_losses.csv'
    if losses.exists():
        loss = table(losses)
        if loss: report.add('TrackingLosses', list(loss), zip(*loss.values()))
        report.metrics['tracking_losses'] = {'count': len(loss.get('start_timestamp', [])), 'observed_duration_s': stats(loss.get('observed_duration_seconds', [])), 'unrecovered': int(np.sum(loss.get('recovered', np.array([])) == 0))}
        if loss:
            report.lines('loss_durations', np.arange(len(loss['recovered'])), {'observed loss duration': loss['observed_duration_seconds']}, 'Duration (s)', 'Loss event index')
    summary = root / 'statistics/summary.txt'
    if summary.exists():
        report.metrics['recorder_summary'] = dict(line.split('=', 1) for line in summary.read_text().splitlines() if '=' in line)
    report.notes += ['tracking_ms 仅包含同步 Track 调用，不含 ROS 转换、异步建图和闭环；1000/tracking_ms 不是系统实际 FPS。',
                     '状态：-1 未准备，0 无图像，1 初始化，2 OK，3 暂时丢失，4 丢失，5 OK_KLT。特征关联比例仅作诊断，不等同于优化器内点率。']


def accuracy(report, stem, t, p, r, gt, args):
    gt_t, gp, gr = gt
    pairs = associate(t+args.time_offset, gt_t, args.max_time_diff)
    if len(pairs) < 3:
        report.metrics[stem]['accuracy_unavailable'] = 'fewer than 3 timestamp matches with ground truth'
        report.notes.append(f'{stem}: 真值匹配不足 3 个，保留几何和性能数据，误差留空。')
        return None
    ii, jj = pairs.T
    try:
        s, a, b = align(p[ii], gp[jj], args.align)
    except ValueError as exc:
        report.metrics[stem]['accuracy_unavailable'] = str(exc)
        report.notes.append(f'{stem}: 无法确定对齐变换：{exc}。误差留空。')
        return None
    aligned, ar = s*(p @ a.T)+b, a @ r
    ep, er, g, rot = aligned[ii], ar[ii], gp[jj], gr[jj]
    error = ep-g
    ate = np.linalg.norm(error, axis=1)
    angles = np.degrees(Rotation.from_matrix(rot.transpose(0, 2, 1) @ er).magnitude())
    m = dict(matched_poses=len(ii), estimated_coverage=len(ii)/len(t), gt_coverage=len(jj)/len(gt_t), alignment=args.align,
             scale=s, rotation=a.tolist(), translation=b.tolist(), timestamp_residual_s=stats(t[ii]+args.time_offset-gt_t[jj]),
             ate_m=stats(ate), absolute_orientation_deg=stats(angles))
    report.metrics[stem]['accuracy'] = m
    report.metrics[stem]['aligned_path_length_m'] = float(s*np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    report.lines(stem+'_aligned_position', t-t[0], {k: aligned[:, n] for n, k in enumerate('XYZ')}, 'Position (m)')
    if len(t) > 1:
        report.lines(stem+'_aligned_speed', t[1:]-t[0], {'estimated speed': s*np.linalg.norm(np.diff(p, axis=0), axis=1)/np.diff(t)}, 'Speed (m/s)')
    plot_trajectory(report, stem+'_aligned_error', ep, g, ate)
    report.lines(stem+'_ate', t[ii]-t[0], {'ATE': ate, **{f'error {k}': error[:, n] for n, k in enumerate('XYZ')}}, 'Error (m)')
    report.distribution(stem+'_ate_distribution', ate, 'ATE (m)')
    report.lines(stem+'_orientation_error', t[ii]-t[0], {'absolute orientation error': angles}, 'Angle (deg)')
    report.csv(stem+'_absolute_errors', ['timestamp', 'gt_timestamp', 'error_x_m', 'error_y_m', 'error_z_m', 'ate_m', 'orientation_deg'], zip(t[ii], gt_t[jj], *error.T, ate, angles))
    # Fixed elapsed-time RPE: never silently interpret matched-row steps as input frames.
    rp = associate(t[ii]+args.rpe_seconds, t[ii], args.rpe_tolerance)
    rp = rp[rp[:, 1] > rp[:, 0]]
    if len(rp):
        values = np.array([relative_error(ep, er, g, rot, i, j) for i, j in rp])
        m['rpe'] = dict(target_interval_s=args.rpe_seconds, tolerance_s=args.rpe_tolerance, actual_interval_s=stats(t[ii[rp[:, 1]]]-t[ii[rp[:, 0]]]), translation_m=stats(values[:, 0]), rotation_deg=stats(values[:, 1]))
        report.lines(stem+'_rpe_translation', t[ii[rp[:, 0]]]-t[0], {'RPE translation': values[:, 0]}, 'Error (m)')
        report.lines(stem+'_rpe_rotation', t[ii[rp[:, 0]]]-t[0], {'RPE rotation': values[:, 1]}, 'Error (deg)')
        report.distribution(stem+'_rpe_distribution', values[:, 0], 'RPE translation (m)')
        report.csv(stem+'_relative_errors', ['start_timestamp', 'end_timestamp', 'translation_m', 'rotation_deg'], zip(t[ii[rp[:, 0]]], t[ii[rp[:, 1]]], *values.T))
    else:
        report.notes.append(f'{stem}: 无满足时间间隔的 RPE 配对。')
    # KITTI-style distances from full GT, evaluate only exact matched endpoints.
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))]
    lookup = dict(zip(jj.tolist(), range(len(jj))))
    segments = []
    for start in range(0, len(gt_t), 10):
        if start not in lookup: continue
        for length in range(100, 801, 100):
            end = int(np.searchsorted(distance, distance[start]+length, side='right'))
            if end not in lookup: continue
            et, ea = relative_error(ep, er, g, rot, lookup[start], lookup[end])
            segments.append((start, end, length, length/(gt_t[end]-gt_t[start]), 100*et/length, ea/length))
    m['distance_segments'] = {'count': len(segments), 'translation_percent': stats([x[4] for x in segments]), 'rotation_deg_per_m': stats([x[5] for x in segments])}
    if segments:
        seg = np.array(segments)
        report.csv(stem+'_distance_segments', ['gt_start_index', 'gt_end_index', 'target_length_m', 'speed_m_s', 'translation_percent', 'rotation_deg_per_m'], segments)
        report.add(stem+'_segments_by_length', ['length_m', 'count', 'translation_mean_percent', 'translation_std_percent', 'rotation_mean_deg_per_m', 'rotation_std_deg_per_m'],
                   ([l, int(np.sum(seg[:, 2] == l)), float(seg[seg[:, 2] == l, 4].mean()), float(seg[seg[:, 2] == l, 4].std()), float(seg[seg[:, 2] == l, 5].mean()), float(seg[seg[:, 2] == l, 5].std())] for l in np.unique(seg[:, 2])),
                   'Line / dispersion bars', 'length_m', 'translation_mean_percent; rotation_mean_deg_per_m')
        for n, ylabel in ((4, 'Translation drift (%)'), (5, 'Rotation drift (deg/m)')):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            lengths = np.unique(seg[:, 2])
            axes[0].plot(lengths, [seg[seg[:, 2] == l, n].mean() for l in lengths], 'o-')
            axes[0].set(xlabel='Segment length (m)', ylabel=ylabel)
            axes[1].scatter(seg[:, 3], seg[:, n], s=6, alpha=.35)
            axes[1].set(xlabel='Segment speed (m/s)', ylabel=ylabel)
            for ax in axes: ax.grid(alpha=.25)
            report.save(stem+('_segment_translation' if n == 4 else '_segment_rotation'), fig)
    report.notes.append(f'{stem}: ATE 使用 {args.align} 对齐；距离分段采用 100–800 m、起点每 10 个真值帧，仅统计两端均匹配的段，为 KITTI 风格诊断，非官方排行榜结果。')
    return s, a, b


def analyze_map(report, stem, points, p, args, transform, reference):
    if not len(points):
        report.notes.append(f'{stem}: 地图点为空。'); return
    m = report.metrics.setdefault(stem, {})
    m['cloud'] = dict(count=len(points), min=points.min(0).tolist(), max=points.max(0).tolist())
    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), min(args.plot_points, len(points)), replace=False)
    sample = points[idx]
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(*sample.T, c=sample[:, 1], s=.2, cmap='viridis', rasterized=True)
    ax.set(xlabel='X', ylabel='Y', zlabel='Z', title=f'Sparse map ({len(sample)} displayed / {len(points)})')
    ax = fig.add_subplot(122)
    im = ax.scatter(sample[:, 0], sample[:, 2], c=sample[:, 1], s=.5, cmap='viridis', rasterized=True)
    if p is not None: ax.plot(p[:, 0], p[:, 2], color='red', lw=.7, label='Camera'); ax.legend()
    ax.set(xlabel='X', ylabel='Z'); ax.axis('equal'); fig.colorbar(im, ax=ax, label='Y coordinate')
    report.save(stem+'_sparse_map', fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.hexbin(points[:, 0], points[:, 2], gridsize=90, mincnt=1, bins='log')
    fig.colorbar(im, ax=ax, label='Point count (log scale)'); ax.set(xlabel='X', ylabel='Z', title='Projected map density'); ax.axis('equal')
    report.save(stem+'_map_density', fig)
    if len(points) > 1:
        nn = cKDTree(points).query(sample, k=2)[0][:, 1]
        m['cloud']['sampled_nearest_neighbor_spacing'] = stats(nn)
        report.distribution(stem+'_point_spacing', nn, 'Nearest other point distance (map units)')
    if reference is None: return
    if transform is None and not args.map_same_frame:
        if 'accuracy_unavailable' in m:
            report.notes.append(f'{stem}: 轨迹无法对齐，参考地图误差留空。'); return
        raise ValueError('Map reference requires trajectory GT alignment, or explicit --map-same-frame.')
    s, a, b = transform if transform is not None else (1., np.eye(3), np.zeros(3))
    transformed = s*(points @ a.T)+b
    forward = cKDTree(reference).query(transformed)[0]
    backward = cKDTree(transformed).query(reference)[0]
    precision, recall = float(np.mean(forward <= args.map_threshold)), float(np.mean(backward <= args.map_threshold))
    m['cloud']['reference_error'] = dict(estimated_to_reference_m=stats(forward), reference_to_estimated_m=stats(backward),
        symmetric_mean_distance_m=float((forward.mean()+backward.mean())/2), threshold_m=args.map_threshold,
        precision=precision, recall=recall, fscore=2*precision*recall/(precision+recall) if precision+recall else 0.)
    report.distribution(stem+'_map_accuracy', forward, 'Estimated to reference nearest point (m)')
    report.distribution(stem+'_map_completeness', backward, 'Reference to estimated nearest point (m)')
    thresholds = np.linspace(0, args.map_threshold*5, 100)
    prec = np.searchsorted(np.sort(forward), thresholds, side='right')/len(forward)
    rec = np.searchsorted(np.sort(backward), thresholds, side='right')/len(backward)
    report.lines(stem+'_map_thresholds', thresholds, {'precision': prec, 'recall': rec, 'F-score': np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec), where=prec+rec>0)}, 'Ratio', 'Distance threshold (m)')
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.scatter(transformed[idx, 0], transformed[idx, 2], c=forward[idx], s=1, cmap='turbo', rasterized=True)
    ax.set(xlabel='X (m)', ylabel='Z (m)', title='Map nearest-reference distance'); ax.axis('equal'); fig.colorbar(im, ax=ax, label='Error (m)')
    report.save(stem+'_map_error', fig)
    report.csv(stem+'_map_point_errors', ['x_m', 'y_m', 'z_m', 'nearest_reference_m'], zip(*transformed.T, forward))
    report.notes.append('地图误差使用全量点最近邻距离，无独立 ICP；稀疏点图相对稠密参考的 completeness/recall 会受采样密度、遮挡与参考覆盖范围影响。')


def analyze(root, output, args, gt, reference):
    report = Report(output, args.formats, args.dpi, excel=getattr(args, 'excel', True))
    report.metrics['source'] = str(root)
    report.metrics['options'] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report.metrics['sensor_mode'] = getattr(args, 'sensor_mode', 'unknown')
    report.metrics['recorder_summary'] = recorder_summary(root)
    report.notes.append(f'传感器模式：{report.metrics["sensor_mode"]}；对齐：{args.align}；KITTI 相机：{args.kitti_camera}。')
    report.notes.append('所有轨迹为最终导出的优化后轨迹；速度、加速度与地图点间距是诊断量，不能替代真值误差。单目未校准时，原始长度单位为任意尺度。')
    analyze_frames(root, report)
    trajectories = sorted((root / 'trajectories').glob('map_*_camera_tum.txt'))
    if not trajectories and (root / 'trajectories/camera_tum.txt').exists():
        trajectories = [root / 'trajectories/camera_tum.txt']
    transforms, positions = {}, {}
    for path in trajectories:
        stem = path.name.removesuffix('_camera_tum.txt') if path.name.startswith('map_') else 'map_0'
        if args.map_id is not None and stem != f'map_{args.map_id}': continue
        if not path.stat().st_size:
            report.notes.append(f'{stem}: 空轨迹，跳过。'); continue
        print(f'  Trajectory {stem}', flush=True)
        t, p, r = trajectory(path)
        positions[stem] = p
        lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
        report.metrics[stem] = dict(poses=len(t), start_timestamp=float(t[0]), duration_s=float(t[-1]-t[0]), path_length_map_units=float(lengths.sum()), endpoint_separation_map_units=float(np.linalg.norm(p[-1]-p[0])))
        plot_trajectory(report, stem+'_trajectory', p)
        report.lines(stem+'_position', t-t[0], {k: p[:, n] for n, k in enumerate('XYZ')}, 'Position (map units)')
        report.lines(stem+'_distance', t-t[0], {'cumulative path': np.r_[0, np.cumsum(lengths)]}, 'Distance (map units)')
        if len(t) > 1:
            dt = np.diff(t); speed = lengths/dt
            omega = np.degrees(Rotation.from_matrix(r[:-1].transpose(0, 2, 1) @ r[1:]).magnitude())/dt
            report.metrics[stem]['speed_map_units_s'] = stats(speed)
            report.lines(stem+'_speed', t[1:]-t[0], {'estimated speed': speed}, 'Speed (map units/s)')
            report.distribution(stem+'_speed_distribution', speed, 'Speed (map units/s)')
            report.lines(stem+'_angular_speed', t[1:]-t[0], {'angular speed': omega}, 'Angular speed (deg/s)')
            if len(t) > 2:
                velocity = np.diff(p, axis=0)/dt[:, None]
                acceleration = np.linalg.norm(np.diff(velocity, axis=0)/((dt[1:]+dt[:-1])/2)[:, None], axis=1)
                report.lines(stem+'_acceleration', t[2:]-t[0], {'finite-difference acceleration': acceleration}, 'Acceleration (map units/s²)')
        keyframes = root / 'trajectories' / f'{stem}_keyframes_tum.txt'
        if keyframes.exists() and keyframes.stat().st_size:
            kt, kp, _ = trajectory(keyframes)
            report.metrics[stem]['keyframes'] = len(kt)
            report.lines(stem+'_keyframes', kt-kt[0], {'retained final keyframes': np.arange(1, len(kt)+1)}, 'Cumulative count (final export)')
            report.distribution(stem+'_keyframe_intervals', np.diff(kt), 'Retained keyframe interval (s)')
        if gt is not None:
            transforms[stem] = accuracy(report, stem, t, p, r, gt, args)
    for path in sorted((root / 'maps').glob('map_*_sparse.ply')):
        stem = path.name.removesuffix('_sparse.ply')
        if args.map_id is not None and stem != f'map_{args.map_id}': continue
        print(f'  Sparse map {stem}', flush=True)
        analyze_map(report, stem, cloud(path), positions.get(stem), args, transforms.get(stem), reference)
    if args.map_id is not None and f'map_{args.map_id}' not in report.metrics:
        raise ValueError(f'Requested map ID {args.map_id} has no usable data.')
    if gt is None: report.notes.append('未提供轨迹真值：ATE、RPE 和距离分段误差不可计算。')
    if reference is None: report.notes.append('未提供参考地图：仅分析地图分布，不将点密度或轨迹误差称为建图误差。')
    report.add('MapStatus', ['map_id', 'status', 'reason'],
               ([stem, 'NO_ALIGNMENT' if 'accuracy_unavailable' in metric else ('OK' if 'accuracy' in metric else 'NO_GT'), metric.get('accuracy_unavailable')]
                for stem, metric in report.metrics.items() if stem.startswith('map_') and isinstance(metric, dict)))
    print(f'Base analysis complete: {output} ({len(report.images)} base figures)', flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input', type=Path, help='Experiment directory, directory containing ZIP archives, or one ZIP archive')
    p.add_argument('-o', '--output', type=Path, default=Path('analysis_output'))
    p.add_argument('--kitti-root', type=Path, help='Auto-match each sequence to poses/NN.txt and sequences/NN/{times,calib}.txt')
    p.add_argument('--no-auto-gt', action='store_true', help='Disable local KITTI root discovery from recorder settings_source')
    p.add_argument('--excel', action='store_true', help='Additionally export Excel workbooks (requires openpyxl)')
    p.add_argument('--sequence', help='Explicit sequence ID for a single experiment, e.g. 00')
    p.add_argument('--gt', type=Path, help='TUM or KITTI camera-to-world ground truth')
    p.add_argument('--gt-times', type=Path, help='KITTI times.txt (required for 12-column GT)')
    p.add_argument('--kitti-calib', type=Path, help='Calibration for converting camera-0 GT to color-camera GT')
    p.add_argument('--sensor-mode', choices=('auto', 'mono', 'stereo', 'rgbd'), default='auto', help='Auto-detect recorder sensor_mode; explicit value overrides detection')
    p.add_argument('--kitti-camera', type=int, choices=(0, 1, 2, 3), help='Auto-detect from camera.yaml KITTI P0/P2 comment; legacy fallback 0')
    p.add_argument('--align', choices=('auto', 'se3', 'sim3', 'none'), default='auto', help='auto: Sim3 for mono, SE3 for stereo/RGB-D/unknown')
    p.add_argument('--time-offset', type=float, default=0., help='Seconds added to estimated timestamps for association')
    p.add_argument('--max-time-diff', type=float, default=.02)
    p.add_argument('--rpe-seconds', type=float, default=1.)
    p.add_argument('--rpe-tolerance', type=float, default=.05, help='Allowed deviation from the target RPE interval (seconds)')
    p.add_argument('--map-reference', type=Path, help='ASCII PLY or XYZ reference in GT world coordinates, meters')
    p.add_argument('--map-same-frame', action='store_true', help='Assert map and reference share metric coordinates when no GT alignment exists')
    p.add_argument('--map-threshold', type=float, default=.2, help='Map precision/recall distance threshold in meters')
    p.add_argument('--map-id', type=int, help='Analyze only this exported map ID')
    p.add_argument('--plot-points', type=int, default=30000, help='Display sample limit; map accuracy still uses all points')
    p.add_argument('--formats', nargs='+', choices=('png', 'tiff', 'pdf', 'svg'), default=['png'])
    p.add_argument('--dpi', type=int, default=1200, help='Publication raster resolution, minimum 1200 DPI')
    return p


def sequence_id(run, override=None):
    if override is not None:
        if not re.fullmatch(r"(?:0?[0-9]|10)", override):
            raise ValueError('--sequence must be 00 through 10')
        return f'{int(override):02d}'
    matches = set()
    for part in run.parts:
        match = re.fullmatch(r'(?:kitti[_-])?(0[0-9]|10)(?:[_-].*)?', part, flags=re.I)
        if match: matches.add(match.group(1))
    settings = recorder_summary(run).get('settings_source', '')
    match = re.fullmatch(r'kitti[_-](0[0-9]|10)(?:[_-].*)?\.(?:yaml|yml)', settings.replace('\\', '/').rsplit('/', 1)[-1], flags=re.I)
    if match: matches.add(match.group(1))
    if len(matches) > 1: raise ValueError(f'Ambiguous sequence IDs in path: {run}')
    return next(iter(matches), None)


def recorder_summary(root):
    path = root/'statistics/summary.txt'
    return dict(line.split('=', 1) for line in path.read_text().splitlines() if '=' in line) if path.exists() else {}


def resolve_options(args, run):
    """Resolve per-run settings, retaining requested options for reproducibility."""
    local = argparse.Namespace(**vars(args))
    summary = recorder_summary(run)
    detected = summary.get('sensor_mode', '').strip().lower()
    if not detected:
        modes = {part.lower().split('_')[-1].split('-')[-1] for part in run.parts} & {'mono', 'stereo', 'rgbd'}
        if len(modes) > 1: raise ValueError(f'Ambiguous sensor modes: {run}')
        detected = next(iter(modes), 'unknown')
    if detected not in ('mono', 'stereo', 'rgbd', 'unknown') and args.sensor_mode == 'auto':
        raise ValueError(f'Unsupported recorder sensor_mode={detected}; expected mono/stereo/rgbd.')
    local.sensor_mode = detected if args.sensor_mode == 'auto' else args.sensor_mode
    local.requested_sensor_mode, local.requested_align = args.sensor_mode, args.align
    local.align = ('sim3' if local.sensor_mode == 'mono' else 'se3') if args.align == 'auto' else args.align
    local.requested_kitti_camera = args.kitti_camera
    if args.kitti_camera is None:
        camera = run/'config/camera.yaml'
        # These comments are written by this project's prepare_kitti_dataset.py.
        text = camera.read_text() if camera.exists() else ''
        comments = re.findall(r'^\s*#.*\bKITTI\b.*$', text, flags=re.M | re.I)
        cameras = {c for comment in comments for c in re.findall(r'\bP([0-3])\b', comment)}
        # Recorder stereo/RGB-D configs describe the pair P2/P3; poses are LEFT camera.
        if cameras in ({'0', '1'}, {'2', '3'}): cameras = {str(min(map(int, cameras)))}
        if len(cameras) > 1: raise ValueError('Ambiguous KITTI camera comments; set --kitti-camera.')
        local.kitti_camera = int(next(iter(cameras))) if cameras else 0
        local.kitti_camera_source = 'camera.yaml KITTI comment' if cameras else 'legacy fallback 0; set --kitti-camera if needed'
    else:
        local.kitti_camera_source = 'explicit --kitti-camera'
    if not args.kitti_root and not args.gt and not getattr(args, 'no_auto_gt', False):
        settings = summary.get('settings_source', '')
        if settings:
            candidate = Path(settings).parent.parent
            seq = sequence_id(run, args.sequence)
            if seq and all((candidate/p).is_file() for p in
                           (f'poses/{seq}.txt', f'sequences/{seq}/times.txt', f'sequences/{seq}/calib.txt')):
                local.kitti_root = candidate
                local.groundtruth_source = 'local KITTI root inferred from recorder settings_source'
    return local


def load_groundtruth(args, seq):
    if args.kitti_root:
        if seq is None: raise ValueError('Cannot identify KITTI sequence. Use kitti_00_* directory names, or --sequence for one run.')
        args.gt = args.kitti_root/'poses'/f'{seq}.txt'
        args.gt_times = args.kitti_root/'sequences'/seq/'times.txt'
        args.kitti_calib = args.kitti_root/'sequences'/seq/'calib.txt'
    gt = trajectory(args.gt, args.gt_times) if args.gt else None
    if args.kitti_camera and gt is not None:
        if gt is None or args.kitti_calib is None:
            raise ValueError('--kitti-camera requires GT and calibration; use --kitti-root or --gt/--kitti-calib.')
        calib = {line.split(':')[0]: np.fromstring(line.split(':')[1], sep=' ') for line in args.kitti_calib.read_text().splitlines() if ':' in line}
        projection = calib[f'P{args.kitti_camera}'].reshape(3, 4)
        center = -np.linalg.solve(projection[:, :3], projection[:, 3])
        gt = (gt[0], gt[1]+gt[2] @ center, gt[2])
    return gt


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.max_time_diff, args.rpe_seconds, args.rpe_tolerance, args.map_threshold, args.plot_points) <= 0:
        raise ValueError('Tolerance, interval, threshold and point limit must be positive.')
    if args.dpi < 1200: raise ValueError('--dpi must be at least 1200.')
    if not np.isfinite([args.max_time_diff, args.rpe_seconds, args.rpe_tolerance, args.map_threshold, args.time_offset]).all():
        raise ValueError('Numeric options must be finite.')
    if args.gt and args.kitti_root: raise ValueError('Use either --gt or --kitti-root, not both.')
    args.formats = list(dict.fromkeys(['png']+args.formats))
    reference = cloud(args.map_reference) if args.map_reference else None
    if reference is not None and not len(reference): raise ValueError('Reference cloud is empty.')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory is not empty; choose a new -o path: {output}')
    with tempfile.TemporaryDirectory(prefix='slam_analysis_') as tmp:
        runs, failures = collect_inputs(args.input, Path(tmp), exclude=output)
        if not runs: raise ValueError('No experiment directories found (expected trajectories/ or statistics/frames.csv).')
        if len(runs) > 1 and (args.gt is not None or reference is not None or args.sequence is not None):
            raise ValueError('For multiple runs, use --kitti-root with sequence directories; explicit --gt/--sequence/--map-reference require one run.')
        if args.excel: runtime()
        reports = []
        output.mkdir(parents=True, exist_ok=True)
        print(f'Discovered {len(runs)} experiments; {len(failures)} archive errors.', flush=True)
        for index, (run, provenance) in enumerate(runs):
            seq = None
            dest = output / f'{index:03d}_{run.name}'
            try:
                seq = sequence_id(run, args.sequence)
                local = resolve_options(args, run)
                gt = load_groundtruth(local, seq)
                print(f'Analyzing {local.sensor_mode} sequence {seq or "unknown"} ({local.align}, camera {local.kitti_camera}): {provenance}', flush=True)
                report = analyze(run, dest, local, gt, reference)
                report.metrics['sequence'] = seq
                report.metrics['source'] = provenance
                report.notes.append(f'序列编号：{seq or "未识别"}。高清图片实际 DPI：{report.dpi}。')
                run_plots(report)
                report.finish(f'{local.sensor_mode.upper()} · sequence {seq or "unknown"} · {run.name}')
                reports.append(report)
                print(f'Saved report: {dest / "report.html"} ({len(report.images)} figures)', flush=True)
            except (ValueError, KeyError, OSError) as e:
                plt.close('all')
                failures.append({'sequence': seq, 'sensor_mode': recorder_summary(run).get('sensor_mode', 'unknown'), 'run': provenance, 'error': str(e)})
                print(f'ERROR: {run.name}: {e}', file=sys.stderr)
        overview = comparison(reports, failures, output, args.formats, args.dpi, excel=args.excel, charts=True)
        with (output / 'report.html').open('a') as f:
            f.write('<h2>实验报告</h2><ul>'+''.join(f'<li><a href="{html.escape(r.path.name)}/report.html">{html.escape(r.metrics["sensor_mode"].upper())} · sequence {r.metrics["sequence"] or "unknown"} · {html.escape(r.path.name)}</a></li>' for r in reports)+'</ul>')
            f.write('<pre>'+html.escape(json.dumps(failures, ensure_ascii=False, indent=2))+'</pre>' if failures else '')
        print(f'Overview: {output / "report.html"}', flush=True)
        return 1 if failures else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
