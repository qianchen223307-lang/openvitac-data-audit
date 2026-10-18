"""Audit the explicit processed ACT contract, with bounded reads."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

DEFAULT_IMAGES = ("cam_high", "tac_left", "tac_right")


def audit_paths(paths: Iterable[str | Path], image_keys=DEFAULT_IMAGES, repeat_threshold=10):
    if repeat_threshold < 2:
        raise ValueError("repeat_threshold must be at least 2")
    image_keys = tuple(image_keys)
    if not image_keys or len(set(image_keys)) != len(image_keys):
        raise ValueError("image_keys must be a nonempty list of unique names")
    if any(not k or "/" in k for k in image_keys):
        raise ValueError("image_keys must be simple dataset names, without slashes")
    diagnostics, episodes = [], []

    def add(check_id, severity, path, message, hint):
        diagnostics.append(dict(check_id=check_id, severity=severity, path=str(path), message=message, hint=hint))

    selected = sorted(Path(p) for p in paths)
    if not selected:
        add("ACT_FILES", "error", "input", "No .hdf5 or .h5 episodes found.", "Pass an episode file or a directory containing processed ACT files.")
    for path in selected:
        episode = {"file": path.name, "frames": 0, "streams": {}, "action_trace": []}
        episodes.append(episode)
        try:
            with h5py.File(path, "r") as f:
                if "embodiment/joint" in f and "action" not in f:
                    add("ACT_FORMAT", "error", path.name, "Raw OpenViTac format is unsupported.", "Run the upstream ACT process_data.py first; this tool does not convert raw data.")
                    continue
                fields = [("action", "vector"), ("observations/qpos", "vector")]
                fields += [(f"observations/images/{name}", "image") for name in image_keys]
                reference_length = None
                for key, kind in fields:
                    field_path = f"{path.name}:/{key}"
                    if key not in f or not isinstance(f[key], h5py.Dataset):
                        add("ACT_REQUIRED", "error", field_path, "Required dataset is missing or is a group.", "Restore this dataset or remove an unneeded image from --images.")
                        continue
                    ds = f[key]
                    shape = tuple(ds.shape)
                    episode["streams"][key] = {"shape": list(shape), "dtype": str(ds.dtype)}
                    if not shape or shape[0] == 0:
                        add("ACT_LENGTH", "error", field_path, "Dataset has no frames.", "Collect or export at least one valid frame.")
                        continue
                    if key == "action":
                        reference_length = shape[0]
                        episode["frames"] = int(reference_length)
                    if reference_length is not None and shape[0] != reference_length:
                        add("ACT_LENGTH", "error", field_path, f"Length {shape[0]} differs from action length {reference_length}.", "Re-export all selected streams using the same valid-pair indices.")
                    valid_shape = (len(shape) == 2 and shape[1] == 8) if kind == "vector" else (len(shape) == 4 and shape[-1] == 3 and min(shape[1:3]) > 0)
                    if not valid_shape:
                        expected = "[T, 8]" if kind == "vector" else "[T, H>0, W>0, 3]"
                        add("ACT_SHAPE", "error", field_path, f"Shape {shape} does not match {expected}.", "Use processed ACT layout; image sizes may differ between streams.")
                    numeric = ds.dtype.kind in "fiu"
                    if kind == "vector" and not numeric:
                        add("ACT_DTYPE", "error", field_path, f"Expected numeric vectors; got {ds.dtype}.", "Export finite numeric state and action values.")
                    if kind == "image" and ds.dtype != np.dtype("uint8"):
                        add("ACT_DTYPE", "error", field_path, f"Expected uint8 images; got {ds.dtype}.", "Export decoded HWC uint8 frames, not JPEG bytes or normalized floats.")
                    run_start, run_length, previous = 0, 0, None
                    repeated, nonfinite = [], []
                    try:
                        for start in range(0, shape[0], 32):
                            block = ds[start:start + 32]
                            if kind == "vector" and numeric:
                                bad_rows = np.flatnonzero(~np.isfinite(block).reshape(len(block), -1).all(axis=1))
                                nonfinite.extend((bad_rows + start).tolist())
                                if key == "action" and valid_shape:
                                    for row in block:
                                        if len(episode["action_trace"]) < 256:
                                            episode["action_trace"].append([float(v) if np.isfinite(v) else None for v in row])
                            if kind == "image" and valid_shape and ds.dtype == np.dtype("uint8"):
                                for offset, frame in enumerate(block):
                                    digest = hashlib.sha256(frame.tobytes()).digest()
                                    idx = start + offset
                                    if digest == previous:
                                        run_length += 1
                                    else:
                                        if run_length >= repeat_threshold:
                                            repeated.append([run_start, idx - 1])
                                        run_start, run_length = idx, 1
                                    previous = digest
                        if run_length >= repeat_threshold:
                            repeated.append([run_start, shape[0] - 1])
                    except (OSError, ValueError, TypeError, RuntimeError) as exc:
                        add("ACT_READ", "error", field_path, f"Dataset cannot be fully read: {exc}", "Re-export the episode and check storage integrity.")
                    if nonfinite:
                        add("ACT_FINITE", "error", field_path, f"NaN/Inf in {len(nonfinite)} frame(s); first indices: {nonfinite[:8]}.", "Inspect acquisition/export; do not silently replace nonfinite robot commands.")
                    if repeated:
                        episode["streams"][key]["repeated_ranges"] = repeated
                        add("ACT_REPEAT", "warning", field_path, f"Identical-frame runs of at least {repeat_threshold} frames: {repeated[:8]}.", "Review these intervals; stationary scenes and no-contact tactile images can legitimately repeat.")
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            add("ACT_OPEN", "error", path.name, f"Cannot open HDF5: {exc}", "Use a readable processed HDF5 file; renaming another format is insufficient.")
    errors = sum(d["severity"] == "error" for d in diagnostics)
    warnings = sum(d["severity"] == "warning" for d in diagnostics)
    return {
        "schema": "openvitac-act-audit-report-v1", "data_kind": "user_input",
        "settings": {"image_keys": list(image_keys), "state_dim": 8, "repeat_threshold": repeat_threshold},
        "summary": {"episodes": len(episodes), "frames": sum(e["frames"] for e in episodes), "errors": errors, "warnings": warnings, "passed": errors == 0},
        "diagnostics": diagnostics, "episodes": episodes,
        "limitations": ["Frame-count agreement is not sensor synchronization verification.", "Repeated images are review warnings, not proof of acquisition failure.", "No simulator, model training, task success, hardware safety, or energy measurements are performed."]}
