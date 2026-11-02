"""Explicit stage commands; demo never silently substitutes for inference."""

import argparse
import json
import sys

from . import pipeline
from .analysis import report
from .config import Config
from .storage import environment


def main(argv=None):
    parser = argparse.ArgumentParser(description="J++ frozen-model recovery pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect installed packages and GPU availability; no downloads")
    prepare = sub.add_parser(
        "prepare", help="Generate data and resolve immutable Hub revisions; no weight downloads"
    )
    prepare.add_argument("--config", default="configs/pilot.toml")
    prepare.add_argument("--run", required=True)
    demo = sub.add_parser("demo", help="Offline simulated end-to-end workflow check, NOT research")
    demo.add_argument("--run", required=True)
    for command in ["collect", "correct", "ablate", "report"]:
        cmd = sub.add_parser(command)
        cmd.add_argument("--run", required=True)
        cmd.add_argument("--split", choices=["dev", "test"], required=True)
        if command != "report":
            cmd.add_argument(
                "--limit", type=int, help="Maximum NEW records this invocation; rerun to resume"
            )
    freeze = sub.add_parser("freeze", help="Lock development and open held-out test")
    freeze.add_argument("--run", required=True)
    freeze.add_argument(
        "--allow-weak-readout",
        metavar="RATIONALE",
        help="Explicitly register a negative-result study if the readout gate fails",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            output = {"packages": environment(), "cuda": False}
            try:
                import torch

                output["cuda"] = torch.cuda.is_available()
                output["gpu"] = torch.cuda.get_device_name(0) if output["cuda"] else None
            except ImportError:
                pass
        elif args.command == "prepare":
            output = pipeline.prepare(Config.load(args.config), args.run)
        elif args.command == "collect":
            output = pipeline.collect(args.run, args.split, args.limit)
        elif args.command in {"correct", "ablate"}:
            output = pipeline.correct(
                args.run,
                args.split,
                limit=args.limit,
                stage="ablation" if args.command == "ablate" else "correction",
            )
        elif args.command == "report":
            output = {"report": report(args.run, args.split)}
        elif args.command == "freeze":
            output = pipeline.freeze(args.run, args.allow_weak_readout)
        else:
            config = Config()
            config.data.dev_count = config.data.test_count = 30
            config.analysis.bootstrap_samples = 200
            pipeline.prepare(config, args.run, demo=True)
            for split in ["dev", "test"]:
                pipeline.collect(args.run, split)
                pipeline.correct(args.run, split)
                pipeline.correct(args.run, split, stage="ablation")
                report(args.run, split)
                if split == "dev":
                    pipeline.freeze(
                        args.run, "Offline simulated workflow check, not a scientific gate"
                    )
            output = {"mode": "DEMO_NOT_RESEARCH", "report": report(args.run, "test")}
        print(json.dumps(output, indent=2))
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
