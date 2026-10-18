import argparse
import json
from pathlib import Path

from .audit import DEFAULT_IMAGES, audit_paths
from .demo import run_demo
from .report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU-only processed OpenViTac ACT HDF5 audit")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Generate deterministic synthetic fixtures and reports")
    demo.add_argument("--root", default=".")
    check = sub.add_parser("check", help="Check a processed HDF5 episode or directory")
    check.add_argument("input", type=Path)
    check.add_argument("--images", default=",".join(DEFAULT_IMAGES), help="Comma-separated required image dataset names")
    check.add_argument("--repeat-threshold", type=int, default=10)
    check.add_argument("--output", default="results/check")
    args = parser.parse_args(argv)
    if args.command == "demo":
        reports = run_demo(args.root)
        print(json.dumps({k: r["summary"] for k, r in reports.items()}, indent=2))
        return 0
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    paths = list(args.input.glob("*.hdf5")) + list(args.input.glob("*.h5")) if args.input.is_dir() else [args.input]
    try:
        report = audit_paths(paths, [k.strip() for k in args.images.split(",")], args.repeat_threshold)
    except ValueError as exc:
        parser.error(str(exc))
    write_report(report, args.output)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
