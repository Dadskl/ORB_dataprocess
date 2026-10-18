# SLAM 实验结果分析

面向 `slam_ros2` 的 `ExperimentRecorder` 输出，批量分析单目、双目和 RGB-D 实验。输入可以是一个实验目录、一个 ZIP，或包含多个实验 ZIP 的目录。程序生成指标 JSON、可追溯的 CSV、HTML 报告和高清图；通过 `--excel` 可额外导出 Excel。绘图由 Python 完成，无需 Origin。

## 环境要求

- Python 3.10 或更新版本
- 用于 ATE、RPE 和距离分段漂移的真值轨迹；没有真值时仍可分析耗时、跟踪状态和地图分布
- 可选：KITTI 参考地图（ASCII PLY 或 XYZ），用于地图精度/召回率

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
# 需要 --excel 时，再安装：
.venv/bin/python -m pip install -r requirements-excel.txt
```

## 快速开始

```bash
.venv/bin/python analyze_results.py /path/to/experiment_archives \
  --kitti-root /path/to/kitti \
  -o /path/to/output
```

第一个参数是输入路径。程序递归查找 ZIP 和原始实验目录；每个 ZIP 独立解压，包内路径可以有多层父目录。实验需包含 `trajectories/` 或 `statistics/frames.csv`。不要把同一实验的原始目录和 ZIP 同时放进输入目录，否则会分析两次。暂不递归解压 ZIP 内的 ZIP。

`-o` 默认是当前目录的 `analysis_output`。输出目录必须不存在或为空，程序不会覆盖已有完整报告。多个 ZIP 中有一个损坏时，其他实验仍会处理，但进程以状态码 1 结束，并在 `metrics.json` 中记录失败原因。ZIP 中的越界路径和符号链接会被拒绝。

使用记录器生成结果时，程序会尝试从 `statistics/summary.txt` 中的 `settings_source` 自动定位 KITTI 真值；迁移机器后建议明确传入 `--kitti-root`。单次实验也可手动指定真值：

```bash
.venv/bin/python analyze_results.py /path/to/one_run \
  --sequence 00 \
  --gt /path/to/kitti/poses/00.txt \
  --gt-times /path/to/kitti/sequences/00/times.txt \
  --kitti-calib /path/to/kitti/sequences/00/calib.txt \
  --kitti-camera 2 \
  -o /path/to/single_output
```

真值支持 TUM 的 `timestamp tx ty tz qx qy qz qw` 格式，或 KITTI 的 12 列位姿矩阵；KITTI 格式需配套 `times.txt`。`--no-auto-gt` 可关闭真值自动定位。`--sensor-mode mono|stereo|rgbd`、`--align se3|sim3|none` 可覆盖自动识别与默认对齐。运行 `python analyze_results.py --help` 查看全部参数。

## 指标与图

| 类别 | 输出内容 |
| --- | --- |
| 轨迹与误差 | 每张地图单独对齐；ATE、绝对姿态误差、RPE 平移/旋转、误差时间曲线、逐地图箱线图与 ECDF |
| 距离分段漂移 | 100–800 m 分段平移漂移、旋转漂移及对应曲线；仅在真值与匹配端点充分时产生 |
| 覆盖与跟踪 | 有效位姿比例、估计与真值覆盖率、跟踪状态及丢失事件 |
| 性能 | Track 和左右特征提取耗时、分位数、直方图、ECDF、时间线及特征数关系 |
| 地图 | 稀疏点云、投影密度、关键帧；提供参考地图时额外计算双向最近邻距离、精度、召回率与 F-score |
| 批次对比 | 按相机模式和序列绘制精度、性能、耗时分布图，并导出逐实验/逐地图汇总表 |

单目默认以 Sim(3) 对齐每张独立地图，消除一个全局尺度因子；双目和 RGB-D 默认以 SE(3) 对齐，保留尺度误差。单目对齐后的 ATE 单位为米，但不代表恢复了绝对尺度。时间匹配为最近邻一对一，不插值；默认容差 0.02 秒。RPE 默认间隔 1 秒，允许偏差 0.05 秒。未提供真值时误差指标留空，绝不按零处理。地图点密度是分布诊断，只有提供参考地图并处于同一坐标系时才计算地图评价指标。

Track 耗时是记录器同步调用时间，不包含全部 ROS 回调与异步建图耗时。箱线图采用 1.5 IQR；ECDF 展示经验累计比例。不同序列路线不同，不能直接把它们的绝对时间曲线当作算法排名。

## 输出结构

```text
output/
  report.html                   # 批次报告
  metrics.json                  # 批次指标与失败记录
  figure_manifest.json         # 图片清单和格式
  01_figures_hd/                # 批次对比图
  03_data/                      # ExperimentSummary、RunSummary 等完整 CSV
  000_<run_name>/
    report.html
    metrics.json
    figure_manifest.json
    01_figures_hd/              # 轨迹、误差、性能、地图等图
    03_data/                    # 绘图原始数值与指标表
    02_origin/                  # 兼容旧流程的 CSV；--excel 时含 XLSX
```

默认输出 PNG，最低 1201 DPI；可用 `--formats png tiff pdf svg` 增加格式。`03_data/PlotCatalog.csv` 说明各表的列、单位与对应图形。空单元格表示指标不可用。批次 `ExperimentSummary.csv` 每实验一行，性能不因多地图重复计数；`RunSummary.csv` 每地图一行。运行 `--excel` 时还生成单次实验的 `02_origin/origin_data.xlsx` 和批次的 `02_origin/sequence_comparison.xlsx`。

## 导出某张图的数据

已生成报告后，可为每张图单独打包原始数据：

```bash
.venv/bin/python export_plot_data.py /path/to/output \
  -o /path/to/plot_data

# 可按图名和实验目录筛选
.venv/bin/python export_plot_data.py /path/to/output \
  --figure 'error_timeline' --run '000_*'
```

`index.csv` 记录图名、相机模式、序列、地图编号和数据包路径。每个 `*.plot-data.zip` 包含对应 CSV、元数据和 SHA-256 校验信息。涉及点云的图需要保留原始实验目录或 ZIP，才能把完整 PLY 一起打包。该命令读取现有报告，不重算指标或绘图。

只需根据已有 CSV 重绘诊断曲线和分布图时运行：

```bash
.venv/bin/python redraw_curves.py /path/to/output
```

## 方法与边界

- ATE 为对齐后位置误差的均方根：`sqrt(mean(||p_est - p_gt||²))`。少于 3 个匹配点或几何退化的地图标记为 `NO_ALIGNMENT`。
- RPE 比较间隔内估计与真值的相对位姿；报告平移范数和旋转角。没有足够配对时留空。
- 距离分段基于完整真值累计距离，每 10 个真值帧取一个起点，仅两端都匹配的分段进入统计。平移漂移按目标长度归一化，旋转漂移单位为 deg/m。这是 KITTI 风格分析，不等同官方排行榜分数。
- 参考地图支持 ASCII PLY 或 XYZ。`--map-reference /path/to/reference.ply --map-threshold 0.2` 指定参考和阈值；无真值对齐但已共用米制坐标系时还需 `--map-same-frame`。不额外执行 ICP。
- 原始速度和加速度由最终轨迹差分计算。样本标准差只描述样本离散程度，不能当作重复实验的置信区间。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q analyze_results.py diagnostic_plots.py \
  export_plot_data.py input_sources.py origin_export.py redraw_curves.py
```

测试使用临时合成输入，不需要下载原始实验数据。实际批次的数值和图形仍应结合所用数据集与真值检查。
