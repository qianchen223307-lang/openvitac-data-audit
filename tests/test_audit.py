import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from openvitac_data_audit.__main__ import main
from openvitac_data_audit.audit import audit_paths
from openvitac_data_audit.demo import make_episode
from openvitac_data_audit.report import write_report


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.tmp.name)
        self.file = self.root / "episode_0.hdf5"
        make_episode(self.file)

    def tearDown(self):
        self.tmp.cleanup()

    def ids(self, report):
        return {d["check_id"] for d in report["diagnostics"]}

    def test_valid_actual_hdf5(self):
        result = audit_paths([self.file])
        self.assertTrue(result["summary"]["passed"])
        self.assertEqual(result["summary"]["frames"], 32)
        self.assertEqual(len(result["episodes"][0]["action_trace"]), 32)

    def test_combined_damage_is_localized(self):
        make_episode(self.file, broken=True)
        r = audit_paths([self.file])
        self.assertEqual(r["summary"]["errors"], 4)
        self.assertEqual(r["summary"]["warnings"], 1)
        self.assertTrue({"ACT_SHAPE", "ACT_FINITE", "ACT_DTYPE", "ACT_LENGTH", "ACT_REPEAT"} <= self.ids(r))
        self.assertTrue(all(set(d) == {"check_id", "severity", "path", "message", "hint"} for d in r["diagnostics"]))

    def test_raw_format_has_actionable_message(self):
        with h5py.File(self.file, "w") as f:
            f.create_dataset("embodiment/joint", data=np.zeros((4, 8)))
        r = audit_paths([self.file])
        self.assertEqual(self.ids(r), {"ACT_FORMAT"})
        self.assertIn("process_data.py", r["diagnostics"][0]["hint"])

    def test_unselected_tactile_not_required(self):
        with h5py.File(self.file, "a") as f:
            del f["observations/images/tac_left"]
            del f["observations/images/tac_right"]
        self.assertTrue(audit_paths([self.file], image_keys=["cam_high"])["summary"]["passed"])
        self.assertIn("ACT_REQUIRED", self.ids(audit_paths([self.file])))

    def test_missing_selected_dataset(self):
        with h5py.File(self.file, "a") as f:
            del f["observations/qpos"]
        self.assertIn("ACT_REQUIRED", self.ids(audit_paths([self.file])))

    def test_different_stream_resolutions_are_valid(self):
        r = audit_paths([self.file])
        streams = r["episodes"][0]["streams"]
        self.assertNotEqual(streams["observations/images/cam_high"]["shape"], streams["observations/images/tac_left"]["shape"])
        self.assertTrue(r["summary"]["passed"])

    def test_corrupt_container_is_reported(self):
        self.file.write_bytes(b"this is not hdf5")
        self.assertEqual(self.ids(audit_paths([self.file])), {"ACT_OPEN"})

    def test_zero_frames_are_errors(self):
        make_episode(self.file, frames=0)
        self.assertIn("ACT_LENGTH", self.ids(audit_paths([self.file])))

    def test_repetition_at_threshold_is_warning(self):
        with h5py.File(self.file, "a") as f:
            ds = f["observations/images/tac_left"]
            ds[0:10] = ds[0]
        r = audit_paths([self.file])
        self.assertEqual(r["summary"]["errors"], 0)
        self.assertEqual(r["summary"]["warnings"], 1)
        self.assertEqual(r["episodes"][0]["streams"]["observations/images/tac_left"]["repeated_ranges"], [[0, 9]])

    def test_read_failure_is_caught(self):
        with h5py.File(self.file, "a") as f:
            del f["observations/qpos"]
            f.create_dataset("observations/qpos", shape=(32, 8), dtype="float32", external=[("missing-backing-file.bin", 0, h5py.h5f.UNLIMITED)])
        self.assertIn("ACT_READ", self.ids(audit_paths([self.file])))

    def test_report_is_strict_json_and_escaped_html(self):
        make_episode(self.file, broken=True)
        r = audit_paths([self.file])
        r["diagnostics"][0]["message"] = "<script>alert('x')</script>"
        out = self.root / "reports"
        write_report(r, out)
        self.assertEqual(json.loads((out / "report.json").read_text())["summary"]["errors"], 4)
        self.assertNotIn("<script>", (out / "report.html").read_text())

    def test_cli_exit_codes_and_real_reports(self):
        self.assertEqual(main(["check", str(self.file), "--output", str(self.root / "valid")]), 0)
        make_episode(self.file, broken=True)
        self.assertEqual(main(["check", str(self.file), "--output", str(self.root / "invalid")]), 1)
        self.assertTrue((self.root / "invalid/report.html").is_file())


if __name__ == "__main__":
    unittest.main()
