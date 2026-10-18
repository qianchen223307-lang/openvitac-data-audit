"""Portable audit reports; HTML is separate from the public portfolio demo."""

import html
import json
from pathlib import Path


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ("check_id", "severity", "path", "message", "hint")
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(d[c]))}</td>" for c in columns) + "</tr>" for d in report["diagnostics"])
    body = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>OpenViTac ACT audit</title><style>body{font:16px system-ui;margin:2rem;max-width:1200px;color:#14362e}table{border-collapse:collapse;width:100%}td,th{border:1px solid #c8d9d2;padding:.7rem;text-align:left;vertical-align:top}pre{white-space:pre-wrap}h1{font-size:2rem}</style><h1>OpenViTac ACT data audit</h1>"
    body += "<p>" + html.escape(report["data_kind"]) + " · CPU-only · schema: openvitac-act-v1</p><pre>" + html.escape(json.dumps(report["summary"], indent=2)) + "</pre>"
    body += "<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr></thead><tbody>" + (rows or "<tr><td colspan='5'>No diagnostics.</td></tr>") + "</tbody></table><p>" + " ".join(html.escape(x) for x in report["limitations"]) + "</p></html>"
    (output / "report.html").write_text(body, encoding="utf-8")
