"""Command-line entrypoints. No model download happens on --help or demo."""

import argparse
import json
import os
from pathlib import Path


def parser():
    cli = argparse.ArgumentParser(description="J-Reflect: readout-trained self-correction")
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report installed versions and CUDA availability")
    for name in [
        "prepare",
        "generate",
        "jpp-fit",
        "gate",
        "examples",
        "train",
        "freeze",
        "prepare-test",
        "predict",
        "report",
    ]:
        command = sub.add_parser(name)
        command.add_argument("--run", required=True, type=Path)
        if name == "prepare":
            command.add_argument("--config", default="configs/initial.toml", type=Path)
        elif name == "generate":
            command.add_argument(
                "--split", choices=["train", "dev_tune", "dev_gate", "test"], required=True
            )
            command.add_argument("--lens", type=Path)
            command.add_argument("--limit", type=int)
        elif name == "jpp-fit":
            command.add_argument("--upstream", type=Path, default=Path("vendor/jpp_lens"))
            command.add_argument(
                "--stage",
                choices=["all", "router", "jacobians", "weights", "validate"],
                default="all",
            )
        elif name in {"train", "predict"}:
            command.add_argument(
                "--arm", choices=["readout", "no_readout", "shuffled"], required=True
            )
            command.add_argument("--seed", type=int, required=True)
            if name == "train":
                command.add_argument(
                    "--max-steps",
                    type=int,
                    help="Stop a pilot after this many optimizer steps; resume by omitting",
                )
            else:
                command.add_argument("--split", choices=["dev_tune", "test"], required=True)
                command.add_argument("--ablation", choices=["readout_removed", "tuned_first"])
        elif name == "report":
            command.add_argument("--split", choices=["dev_tune", "test"], default="test")
    for name in ["demo", "smoke"]:
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, default=Path("artifacts") / name)
        if name == "smoke":
            command.add_argument("--upstream", type=Path)
    return cli


def execute(args):
    if args.command == "doctor":
        from .artifacts import provenance

        result = provenance()
        try:
            import torch

            result["cuda_available"] = torch.cuda.is_available()
            result["cuda_runtime"] = torch.version.cuda
            result["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
        except ImportError:
            result["cuda_available"] = False
        return result
    if args.command == "prepare":
        from .config import load_config
        from .data import prepare

        prepare(load_config(args.config), args.run)
        return {"prepared": str(args.run), "test_loaded": False}
    if args.command == "generate":
        from .pipeline import generate_split

        return generate_split(args.run, args.split, lens_path=args.lens, limit=args.limit)
    if args.command == "jpp-fit":
        from .jpp import fit

        return fit(args.run, args.upstream, args.stage)
    if args.command == "gate":
        from .sanity import run_gate

        result = run_gate(args.run)
        return {k: v for k, v in result.items() if k != "prediction_rows"}
    if args.command == "examples":
        from .training import prepare_examples

        return prepare_examples(args.run)
    if args.command == "train":
        from .training import train

        return train(args.run, args.arm, args.seed, max_steps=args.max_steps)
    if args.command == "freeze":
        from .protocol import freeze

        return freeze(args.run)
    if args.command == "prepare-test":
        from .data import prepare_test

        return {"test_items": len(prepare_test(args.run))}
    if args.command == "predict":
        from .inference import predict

        return {
            "predictions": predict(
                args.run, args.split, args.arm, args.seed, ablation=args.ablation
            )
        }
    if args.command == "report":
        from .data import load_run
        from .evaluation import evaluate_predictions
        from .inference import collect_predictions
        from .reporting import write_report
        from .sanity import checked_base

        _, config = load_run(args.run)
        _, base = checked_base(args.run, args.split)
        result = evaluate_predictions(
            base,
            collect_predictions(args.run, args.split, base, config.training.seeds),
            config.training.seeds,
            bootstrap_samples=config.evaluation.bootstrap_samples,
            confidence=config.evaluation.confidence,
            seed=config.seed,
        )
        return {"report": write_report(args.run / "reports" / args.split, result, split=args.split)}
    if args.command == "demo":
        from .diagnostics import demo

        return {"synthetic_report": demo(args.output)}
    if args.command == "smoke":
        from .diagnostics import smoke

        return smoke(args.output, args.upstream)
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    cache = Path.cwd() / ".cache"
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        print(json.dumps(execute(args), indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser().exit(2, f"jreflect: {exc}\n")


if __name__ == "__main__":
    main()
