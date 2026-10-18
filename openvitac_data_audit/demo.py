"""Deterministic, independently generated geometry fixtures and results."""

import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from .audit import DEFAULT_IMAGES, audit_paths
from .report import write_report


def make_episode(path, broken=False, frames=32):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(frames, dtype=np.float32)
    qpos = np.stack([np.sin(t / 8 + k / 3) * .15 + k / 10 for k in range(8)], axis=1)
    action = np.stack([np.sin((t + 1) / 8 + k / 3) * .15 + k / 10 for k in range(8)], axis=1)
    with h5py.File(path, "w") as f:
        f.attrs["source_phase"] = "action"
        f.attrs["data_kind"] = "synthetic_example"
        if broken and frames > 6:
            action[6, 2] = np.nan
        f.create_dataset("action", data=action)
        f.create_dataset("observations/qpos", data=qpos[:, :7] if broken else qpos)
        for key in DEFAULT_IMAGES:
            size = 64 if key == "cam_high" else 32
            images = []
            for i in range(frames):
                img = Image.new("RGB", (size, size), (21, 43, 50))
                draw = ImageDraw.Draw(img)
                x = 3 + i % max(1, size - 16)
                draw.rectangle((x, 8, x + 10, 24), fill=(20 + i * 5 % 200, 180, 130))
                draw.line((0, (i * 3) % size, size - 1, (i * 3) % size), fill=(240, 180, 60))
                images.append(np.asarray(img))
            images = np.stack(images) if images else np.empty((0, size, size, 3), dtype=np.uint8)
            if broken and key == "tac_left" and frames >= 24:
                images[10:24] = images[10]
            if broken and key == "tac_right":
                images = images[:-1]
            if broken and key == "cam_high":
                images = images.astype(np.float32) / 255
            f.create_dataset(f"observations/images/{key}", data=images, compression="gzip")


def _visuals(valid_path, output):
    output.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (720, 260), "#edf3ef")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), "SYNTHETIC GEOMETRY / processed ACT streams / frame 12", fill="#163d34")
    with h5py.File(valid_path, "r") as f:
        for i, name in enumerate(DEFAULT_IMAGES):
            frame = Image.fromarray(f[f"observations/images/{name}"][12]).resize((192, 192))
            canvas.paste(frame, (20 + i * 236, 40))
            draw.text((20 + i * 236, 237), name, fill="#163d34")
        values = f["action"][:, 0]
    canvas.save(output / "streams.png")
    chart = Image.new("RGB", (720, 300), "#edf3ef")
    d = ImageDraw.Draw(chart)
    d.text((24, 15), "Synthetic action[0] / 32 frames / no robot hardware", fill="#163d34")
    d.line((50, 50, 50, 250, 690, 250), fill="#6d8c81", width=2)
    lo, hi = float(min(values)), float(max(values))
    points = [(50 + i * 640 / (len(values) - 1), 235 - (float(v) - lo) * 170 / max(hi - lo, 1e-9)) for i, v in enumerate(values)]
    d.line(points, fill="#147d60", width=4)
    d.text((50, 265), f"frame 0                         frame 16                         frame 31   range {lo:.3f}..{hi:.3f}", fill="#163d34")
    chart.save(output / "action-curve.png")


def run_demo(root):
    root = Path(root)
    valid = root / "fixtures/valid/episode_0.hdf5"
    invalid = root / "fixtures/invalid/episode_0.hdf5"
    raw = root / "fixtures/raw/episode_0.hdf5"
    make_episode(valid)
    make_episode(invalid, broken=True)
    raw.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw, "w") as f:
        f.create_dataset("embodiment/joint", data=np.zeros((32, 8), dtype=np.float32))
    reports = {}
    for label, path in (("valid", valid), ("invalid", invalid), ("raw", raw)):
        report = audit_paths([path])
        report["data_kind"] = "synthetic_demo"
        write_report(report, root / "results" / label)
        reports[label] = report
    _visuals(valid, root / "docs/assets")
    (root / "results/summary.json").write_text(json.dumps({k: v["summary"] for k, v in reports.items()}, indent=2) + "\n", encoding="utf-8")
    valid_trace = reports["valid"]["episodes"][0]["action_trace"]

    def scenario(key, label, summary):
        r = reports[key]
        return {"id": key, "label": label, "summary": summary,
                "metrics": [{"label": "错误", "value": r["summary"]["errors"]}, {"label": "提示", "value": r["summary"]["warnings"]}, {"label": "帧数", "value": r["summary"]["frames"]}],
                "charts": [], "tables": [{"title": "检查诊断", "columns": ["检查", "级别", "路径", "说明"], "rows": [[d["check_id"], d["severity"], d["path"], d["message"]] for d in r["diagnostics"]], "note": "重复图像只提示复查；长度一致不代表传感器同步。"}]}
    data = {"slug": "openvitac-data-audit", "title": "OpenViTac 多模态数据质检", "tagline": "先核对数据，再启动训练", "category": "研究工程工具", "data_kind": "synthetic_demo",
            "summary": "独立编写的 CPU 质检工具，读取 ACT 处理后 HDF5，定位字段、帧数、数值和图像问题；演示全部使用自建几何数据。",
            "metrics": [{"label": "正常样例错误", "value": reports["valid"]["summary"]["errors"]}, {"label": "异常样例错误", "value": reports["invalid"]["summary"]["errors"]}, {"label": "检测图像流", "value": 3}, {"label": "状态/动作维数", "value": 8}],
            "charts": [{"title": "自建动作序列 · 前两个维度", "kind": "line", "labels": list(range(len(valid_trace))), "series": [{"name": f"action[{j}]", "values": [r[j] for r in valid_trace]} for j in range(2)], "y_label": "合成关节位置"}],
            "tables": [{"title": "支持的数据布局", "columns": ["路径", "约束"], "rows": [["/action", "[T,8] finite numeric"], ["/observations/qpos", "[T,8] finite numeric"], ["/observations/images/{name}", "[T,H,W,3] uint8"]]}],
            "images": [{"src": "assets/streams.png", "caption": "独立生成的几何图像 · 三路数据预览 · 并非传感器采集"}, {"src": "assets/action-curve.png", "caption": "实际合成 action[0] 数组曲线 · 未运行机器人"}],
            "scenarios": [scenario("valid", "正常样例", "32 帧、三路图像和 8 维状态/动作，所有硬性检查通过。"), scenario("invalid", "注入四类错误", "错误维数、NaN、图像 dtype 和流长度不一致；另有重复触觉帧提示。"), scenario("raw", "原始格式", "识别原始 embodiment/joint 布局并明确要求先转换。")],
            "method": ["先确认 processed ACT 布局与选中的图像字段。", "按 32 帧块实际读取，检查 T、shape、dtype、有限值和图像重复区间。", "每条诊断记录 check_id / severity / path / message / hint；CLI 返回可用于 CI 的退出码。"],
            "limitations": reports["valid"]["limitations"] + ["不读取 raw OpenViTac、LeRobot 或任意自定义 HDF5；未运行机器人或复现论文性能。"]}
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs/data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return reports
