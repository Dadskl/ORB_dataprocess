"""Package the exact source samples for each rendered SLAM figure.

The analyzer already writes full numeric CSV tables to 03_data. This tool
connects each figure to its tables and restores cloud points from the recorded
source archive, so a single small ZIP is enough to discuss one figure later.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


MAP_FIGURE = re.compile(r"^(map_\d+)_(trajectory|aligned_error|sparse_map|map_density|map_error|segment_translation|segment_rotation)$")
ERROR_PAGE = re.compile(r"^error_distributions_(\d+)$")
MOTION_SUFFIXES = ("aligned_speed", "angular_speed", "speed", "acceleration")


def map_stems(metrics):
    return sorted((key for key, value in metrics.items()
                   if re.fullmatch(r"map_\d+", key) and isinstance(value, dict)),
                  key=lambda key: int(key.split("_")[1]))


def has_rows(data_dir, name):
    path = data_dir / f"{name}.csv"
    if not path.is_file():
        return False
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        return next(reader, None) is not None


def figure_sources(figure, metrics, data_dir):
    """Return CSV table keys and optional source cloud for one figure."""
    match = MAP_FIGURE.fullmatch(figure)
    if match:
        stem, kind = match.groups()
        if kind == "trajectory":
            return [stem + "_trajectory_xyz"], None
        if kind == "aligned_error":
            return [stem + "_aligned_error_xyz"], None
        if kind == "sparse_map":
            return [stem + "_trajectory_xyz"], stem
        if kind == "map_density":
            return [], stem
        if kind == "map_error":
            return [stem + "_map_point_errors"], None
        return [stem + "_distance_segments", stem + "_segments_by_length"], None
    if figure in ("timing_vs_features", "performance_timeline"):
        return ["Frames"], None
    if figure == "performance_distributions":
        return ["Frames", "TrackingLosses"], None
    if figure in ("error_timeline", "position_residuals"):
        available = [stem for stem in map_stems(metrics) if has_rows(data_dir, stem + "_absolute_errors")]
        return ([stem + "_absolute_errors" for stem in available] +
                ([stem + "_relative_errors" for stem in available] if figure == "error_timeline" else [])), None
    page = ERROR_PAGE.fullmatch(figure)
    if page:
        available = [stem for stem in map_stems(metrics) if has_rows(data_dir, stem + "_absolute_errors")]
        group = available[(int(page.group(1)) - 1) * 16:int(page.group(1)) * 16]
        return [name for stem in group for name in (stem + "_absolute_errors", stem + "_relative_errors")], None
    if figure == "motion_diagnostics":
        return [stem + "_" + suffix for stem in map_stems(metrics) for suffix in MOTION_SUFFIXES], None
    if figure == "comparison_performance":
        return ["ExperimentSummary"], None
    if figure == "comparison_accuracy":
        return ["RunSummary"], None
    if figure == "comparison_timing_distributions":
        return ["Track_Box", "ExperimentSummary"], None
    raise ValueError(f"Unknown figure: {figure}")


def cloud_bytes(source, stem, archives):
    name = stem + "_sparse.ply"
    if "::" in source:
        archive_name, inner = source.split("::", 1)
        archive = Path(archive_name)
        if archive not in archives:
            archives[archive] = ZipFile(archive)
        member = str(PurePosixPath(inner) / "maps" / name)
        return archives[archive].read(member), f"{archive}::{member}"
    path = Path(source) / "maps" / name
    return path.read_bytes(), str(path)


def displayed_points(ply, count):
    """Match analyze_map's ASCII PLY order and fixed-seed display sample."""
    stream = io.StringIO(ply.decode("utf-8"))
    if stream.readline().strip() != "ply" or stream.readline().strip() != "format ascii 1.0":
        raise ValueError("Only the analyzer's ASCII PLY format is supported")
    vertex_count = None
    names = []
    in_vertex = False
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PLY header has no end_header")
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            vertex_count, in_vertex = int(parts[2]), True
        elif parts and parts[0] == "element":
            in_vertex = False
        elif parts and parts[0] == "property" and in_vertex:
            names.append(parts[-1])
        elif parts == ["end_header"]:
            break
    if vertex_count is None or not all(axis in names for axis in ("x", "y", "z")):
        raise ValueError("PLY has no XYZ vertices")
    points = np.loadtxt(stream, max_rows=vertex_count, ndmin=2)
    if points.shape != (vertex_count, len(names)):
        raise ValueError("PLY vertex count/properties mismatch")
    points = points[:, [names.index(axis) for axis in ("x", "y", "z")]]
    selected = np.random.default_rng(0).choice(len(points), min(count, len(points)), replace=False)
    return points[selected], vertex_count


def sample_csv(points, headers):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(points)
    return stream.getvalue().encode("utf-8-sig")


def table_info(name, payload):
    rows = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
    headers = next(rows, [])
    return {"name": name, "columns": headers, "rows": sum(1 for _ in rows),
            "sha256": hashlib.sha256(payload).hexdigest()}


def context_for(metrics, figure):
    options = metrics.get("options", {})
    context = {"sensor_mode": metrics.get("sensor_mode"), "sequence": metrics.get("sequence"),
               "source": metrics.get("source"), "alignment": options.get("align"),
               "kitti_camera": options.get("kitti_camera"), "time_offset_s": options.get("time_offset"),
               "association_tolerance_s": options.get("max_time_diff"),
               "rpe_target_s": options.get("rpe_seconds"), "rpe_tolerance_s": options.get("rpe_tolerance")}
    match = MAP_FIGURE.fullmatch(figure)
    if match:
        context["map_id"] = match.group(1)
        context["map_metrics"] = metrics.get(match.group(1))
    else:
        context["map_id"] = None
        if figure.startswith(("error_", "position_", "motion_")):
            context["maps"] = {stem: metrics[stem] for stem in map_stems(metrics)}
        if figure.startswith("performance_") or figure == "timing_vs_features":
            context["frames"] = metrics.get("frames")
            context["tracking_losses"] = metrics.get("tracking_losses")
    return context


def export_one(run_dir, figure, metrics, output, archives, run_name=None):
    data_dir = run_dir / "03_data"
    keys, cloud_stem = figure_sources(figure, metrics, data_dir)
    required = [key for key in keys if key != "TrackingLosses" and not key.endswith("_relative_errors")
                and not key.endswith("_segments_by_length") and not key.endswith(tuple("_" + x for x in MOTION_SUFFIXES))]
    missing = [key for key in required if not (data_dir / f"{key}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir.name}/{figure}: missing required tables: {', '.join(missing)}")
    package = output / (run_name or run_dir.name) / f"{figure}.plot-data.zip"
    package.parent.mkdir(parents=True, exist_ok=True)
    files = []
    with ZipFile(package, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for key in dict.fromkeys(keys):
            path = data_dir / f"{key}.csv"
            if not path.is_file():
                continue  # Optional RPE, loss events, or motion component.
            payload = path.read_bytes()
            name = f"tables/{key}.csv"
            bundle.writestr(name, payload)
            files.append({**table_info(name, payload), "role": "numeric_samples"})
        if cloud_stem:
            ply, source = cloud_bytes(metrics["source"], cloud_stem, archives)
            name = f"cloud/{cloud_stem}_sparse.ply"
            bundle.writestr(name, ply)
            files.append({"name": name, "sha256": hashlib.sha256(ply).hexdigest(),
                          "bytes": len(ply), "role": "all_map_points", "source": source})
            if figure.endswith("_sparse_map"):
                count = int(metrics.get("options", {}).get("plot_points", 30000))
                sample, total = displayed_points(ply, count)
                payload = sample_csv(sample, ["x_map_units", "y_map_units", "z_map_units"])
                name = f"tables/{cloud_stem}_displayed_points.csv"
                bundle.writestr(name, payload)
                files.append({**table_info(name, payload), "role": "displayed_cloud_points",
                              "sample_seed": 0, "full_cloud_points": total})
        if figure.endswith("_map_error"):
            key = figure.removesuffix("_map_error") + "_map_point_errors"
            path = data_dir / f"{key}.csv"
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.reader(stream)
                headers = next(reader)
                points = np.asarray([[float(x) for x in row] for row in reader])
            count = int(metrics.get("options", {}).get("plot_points", 30000))
            idx = np.random.default_rng(0).choice(len(points), min(count, len(points)), replace=False)
            payload = sample_csv(points[idx], headers)
            name = f"tables/{key}_displayed.csv"
            bundle.writestr(name, payload)
            files.append({**table_info(name, payload), "role": "displayed_map_error_points", "sample_seed": 0})
        manifest = {"schema": "slam-plot-data-v1", "figure": figure,
                    "image": f"{run_dir.name}/01_figures_hd/{figure}.png",
                    "image_exists": (run_dir / "01_figures_hd" / f"{figure}.png").is_file(),
                    "context": context_for(metrics, figure), "files": files,
                    "plot_notes": ["CSV 数值为原始样本；不平滑、不插值，缺失值不补零。",
                                   "时间曲线按独立地图绘制；时间间隔大于中位正间隔的 5 倍时断线。",
                                   "分布图的箱线须为 1.5 IQR；各地图或实验的样本不可视为独立重复试验。"]}
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        bundle.writestr("README.txt", "此包保存一张图对应的完整数值样本、口径及来源。请将整个 ZIP 发给分析者，并说明要分析的图名。\n"
                        "查看 manifest.json 获取图名、模式、序列、地图编号、列名、样本数和 SHA-256。\n"
                        "tables/ 为 CSV；cloud/ 为原始 ASCII PLY。缺失的可选数据表示没有符合条件的样本。\n")
    return package, manifest


def export_all(analysis, output, pattern="*", run_pattern="*"):
    analysis, output = Path(analysis).resolve(), Path(output).resolve()
    if not (analysis / "figure_manifest.json").is_file():
        raise ValueError(f"No figure_manifest.json in {analysis}")
    if output == analysis or analysis in output.parents:
        raise ValueError("Output must be outside the analysis directory")
    runs = [analysis] + sorted(path for path in analysis.iterdir()
                               if path.is_dir() and (path / "figure_manifest.json").is_file())
    rows, archives = [], {}
    try:
        for run_dir in runs:
            if not fnmatch.fnmatch("batch" if run_dir == analysis else run_dir.name, run_pattern):
                continue
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            figures = json.loads((run_dir / "figure_manifest.json").read_text(encoding="utf-8"))["figures"]
            for figure in figures:
                if not fnmatch.fnmatch(figure, pattern):
                    continue
                # Place top-level comparison graphs under batch/, all others by run.
                package, manifest = export_one(run_dir, figure, metrics, output, archives,
                                               "batch" if run_dir == analysis else run_dir.name)
                rows.append({"figure": figure, "run": "batch" if run_dir == analysis else run_dir.name,
                             "sensor_mode": manifest["context"]["sensor_mode"],
                             "sequence": manifest["context"]["sequence"],
                             "map_id": manifest["context"]["map_id"],
                             "package": str(package.relative_to(output)),
                             "image_exists": manifest["image_exists"],
                             "tables": "; ".join(f["name"] for f in manifest["files"] if f["name"].endswith(".csv")),
                             "cloud": next((f["name"] for f in manifest["files"] if f["name"].endswith(".ply")), "")})
            if figures and any(fnmatch.fnmatch(fig, pattern) for fig in figures):
                print(f"{run_dir.name}: {sum(row['run'] == ('batch' if run_dir == analysis else run_dir.name) for row in rows)} packages", flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "index.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["figure", "run", "sensor_mode", "sequence", "map_id", "package", "image_exists", "tables", "cloud"])
        writer.writeheader()
        writer.writerows(rows)
    (output / "index.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description="为每张实验图导出独立的原始绘图数据包")
    parser.add_argument("analysis", type=Path, help="analyze_results.py 生成的总报告或单次实验目录")
    parser.add_argument("-o", "--output", type=Path, help="输出目录，默认在输入目录旁边建立 *_plot_data")
    parser.add_argument("--figure", default="*", help="图名筛选，支持通配符，例如 error_timeline 或 map_*_trajectory")
    parser.add_argument("--run", default="*", help="实验目录筛选，batch 表示总报告；支持通配符")
    args = parser.parse_args()
    output = args.output or args.analysis.with_name(args.analysis.name + "_plot_data")
    rows = export_all(args.analysis, output, args.figure, args.run)
    print(f"Exported {len(rows)} plot-data packages to {output}")


if __name__ == "__main__":
    main()
