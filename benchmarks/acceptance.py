from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from htm_code_native.cli import load_config
from htm_code_native.data.types import TrainingPhase
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.training import build_probe_examples, run_phase_exit_probes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--phase")
    parser.add_argument("--probe-set", default="default")
    parser.add_argument("--repo-root")
    parser.add_argument("--report-path", action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    phase = TrainingPhase(args.phase or config.model.training_phase)
    examples = build_probe_examples(
        args.probe_set,
        repo_root=args.repo_root,
        report_paths=args.report_path,
    )
    model = PhaseACodeModel(config)
    report = run_phase_exit_probes(
        model,
        examples,
        config,
        phase,
        probe_set=args.probe_set,
        max_steps=args.max_steps,
    )
    print(
        json.dumps(
            {
                "phase": report.phase,
                "probe_set": report.probe_set,
                "passed": report.passed,
                "metrics": report.metrics,
                "failing_checks": list(report.failing_checks),
                "example_count": report.example_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
