# OpenViTac ACT Data Audit

[Live demo](https://qianchen223307-lang.github.io/openvitac-data-audit/) · [中文](README.md)

An independently implemented, CPU-only checker for **processed OpenViTac ACT HDF5 episodes**. Original contributions: contract checks, CLI, structured JSON/HTML reporting, deterministic synthetic geometry fixtures, tests and a public demo. No upstream implementation was copied. The demo is synthetic and does not demonstrate robot operation or reproduce paper metrics.

```bash
python -m pip install -e .
python -m openvitac_data_audit demo
python -m unittest discover -s tests -v
python -m openvitac_data_audit check /path/to/episodes --images cam_high,tac_left,tac_right --output results/check
```

Input contract: `/action` and `/observations/qpos` are finite numeric `[T,8]`; selected `/observations/images/{name}` datasets are decoded `uint8 [T,H,W,3]`. T must be positive and shared by selected streams. Spatial resolutions may differ. Default streams: `cam_high,tac_left,tac_right`; `cam_wrist` is required only when selected. Files are read-only; directory scans are nonrecursive. Typical raw OpenViTac files receive an explicit unsupported-format diagnostic.

Checks cover missing fields, unreadable containers/datasets, dimensions, dtypes, frame counts and NaN/Inf. Data are checked in 32-frame blocks; action trace output is capped at 256 frames. Exact image repetitions lasting at least 10 frames produce review warnings, because stationary and no-contact images may legitimately repeat.

Each diagnostic has `check_id,severity,path,message,hint`. Results are `report.json` and self-contained `report.html`. Exit codes: 0 = no errors (warnings allowed), 1 = invalid data, 2 = CLI usage/missing input. Demo completion returns 0 while retaining deliberately injected errors in its reports.

Actual generated summaries are in `results/summary.json`: valid 0 errors/0 warnings; damaged 4 errors/1 warning; raw 1 error. Tests run against actual temporary HDF5 files, including missing external backing storage. Previews in `docs/assets/` depict generated geometry, not sensor captures.

Limits: frame-count agreement does not verify synchronization; no latency inference, model training, simulator, robot safety, success-rate or energy measurement. See [SOURCES.md](SOURCES.md) for the pinned interface reference and [AI_USE.md](AI_USE.md) for assistance disclosure.
