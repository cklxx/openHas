#!/usr/bin/env python3
"""AutoRecall — self-improving eval loop.

Chains: eval → export failures → mine hard negatives → retrain CE → re-eval.
Each cycle mines NEW hard negatives from the UPDATED model's failures.
Training set is cumulative: cycle N includes all hard negatives from cycles 0..N-1.

Usage:
    python evals/autorecall.py --cycles 3
    python evals/autorecall.py --cycles 10 --convergence-threshold 0.005
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_SCRIPT_DIR = Path(__file__).parent
_EVAL_SCRIPT = _SCRIPT_DIR / "recall_eval.py"
_GEN_SCRIPT = _SCRIPT_DIR / "gen_rerank_data.py"
_TRAIN_SCRIPT = _SCRIPT_DIR / "train_reranker.py"
_FAILURES_PATH = _SCRIPT_DIR / "failures.jsonl"
_TRAIN_DATA = _SCRIPT_DIR / "rerank_train.jsonl"
_MODEL_DIR = _SCRIPT_DIR / "reranker_model"
_LOG_FILE = _SCRIPT_DIR / "experiment_log.tsv"
_DEAD_ENDS = _SCRIPT_DIR / "dead_ends.jsonl"
_CONVERGENCE_LOG = _SCRIPT_DIR / "convergence.tsv"

_CONVERGENCE_WINDOW = 3
_DEFAULT_THRESHOLD = 0.005
_AR5_KEEP_THRESHOLD = 0.99


@dataclass(frozen=True, slots=True)
class CycleResult:
    cycle_num: int
    r1: float
    r3: float
    mrr: float
    ar5: float
    latency_s: float
    status: Literal['kept', 'discarded']
    hard_neg_count: int
    training_size: int


@dataclass(frozen=True, slots=True)
class _Metrics:
    r1: float
    r3: float
    r5: float
    mrr: float
    ar5: float
    latency_s: float


def _parse_mean_line(output: str) -> _Metrics:
    """Extract metrics from the MEAN line of recall_eval output."""
    for line in output.splitlines():
        if 'MEAN' not in line:
            continue
        parts = line.split()
        idx = parts.index('MEAN') + 1
        r1 = float(parts[idx])
        r3 = float(parts[idx + 1])
        r5 = float(parts[idx + 2])
        mrr = float(parts[idx + 3])
        ar5_match = re.search(r'AR@5=([0-9.]+)', line)
        ar5 = float(ar5_match.group(1)) if ar5_match else 1.0
        lat_match = re.search(r'([0-9.]+)s\s*$', line)
        latency = float(lat_match.group(1)) if lat_match else 0.0
        return _Metrics(r1=r1, r3=r3, r5=r5, mrr=mrr, ar5=ar5, latency_s=latency)
    msg = "Could not find MEAN line in eval output"
    raise ValueError(msg)


def _run_eval(extra_args: list[str] | None = None) -> str:
    """Run recall_eval.py and return stdout."""
    cmd = [
        sys.executable, str(_EVAL_SCRIPT),
        "--hybrid", "--cross-encoder",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout + result.stderr


def _run_gen_data() -> int:
    """Run gen_rerank_data.py and return training row count."""
    subprocess.run(
        [sys.executable, str(_GEN_SCRIPT)], capture_output=True, text=True, check=True,
    )
    with open(_TRAIN_DATA) as f:
        return sum(1 for _ in f)


def _run_train() -> None:
    """Train cross-encoder on current training data."""
    subprocess.run(
        [sys.executable, str(_TRAIN_SCRIPT), "--epochs", "10", "--lr", "5e-5"],
        capture_output=True, text=True, check=True,
    )


def _count_failures() -> int:
    if not _FAILURES_PATH.exists():
        return 0
    with open(_FAILURES_PATH) as f:
        return sum(1 for line in f if line.strip())


def _log_result(result: CycleResult, description: str) -> None:
    """Append to experiment_log.tsv."""
    with open(_LOG_FILE, "a") as f:
        f.write(
            f"cycle-{result.cycle_num}\t{result.r1:.3f}\t{result.r3:.3f}\t"
            f"0.00\t{result.mrr:.3f}\t{result.ar5:.2f}\t{result.latency_s:.1f}s\t"
            f"{result.status}\t{description}\n"
        )


def _log_dead_end(cycle: int, r1: float, delta: float, reason: str) -> None:
    with open(_DEAD_ENDS, "a") as f:
        entry = {
            "cycle": cycle, "r1": r1,
            "r1_delta": delta, "reason": reason,
        }
        f.write(json.dumps(entry) + "\n")


def _log_convergence(cycle: int, r1: float, training_size: int, hard_neg: int) -> None:
    header_needed = not _CONVERGENCE_LOG.exists()
    with open(_CONVERGENCE_LOG, "a") as f:
        if header_needed:
            f.write("cycle\tR@1\ttraining_size\thard_neg_count\n")
        f.write(f"{cycle}\t{r1:.4f}\t{training_size}\t{hard_neg}\n")


def _check_convergence(
    history: list[float], threshold: float,
) -> bool:
    """Return True if last WINDOW deltas are all below threshold."""
    if len(history) < _CONVERGENCE_WINDOW + 1:
        return False
    recent = history[-_CONVERGENCE_WINDOW:]
    prev = history[-(1 + _CONVERGENCE_WINDOW):-1]
    return all(abs(r - p) < threshold for r, p in zip(recent, prev, strict=True))


def _eval_and_mine(cycle_num: int) -> tuple[_Metrics, int, int]:
    """Eval current model, export failures, generate training data."""
    print("  Step 1: Eval + export failures...")
    output = _run_eval(["--export-failures", str(_FAILURES_PATH)])
    metrics = _parse_mean_line(output)
    hard_neg = _count_failures()
    print(f"    R@1={metrics.r1:.3f} AR@5={metrics.ar5:.2f} failures={hard_neg}")
    print("  Step 2: Generating training data...")
    training_size = _run_gen_data()
    print(f"    Training rows: {training_size}")
    return metrics, hard_neg, training_size


def _retrain_and_reeval() -> _Metrics:
    """Retrain CE and re-evaluate."""
    print("  Step 3: Retraining cross-encoder...")
    _run_train()
    print("  Step 4: Re-evaluating...")
    output = _run_eval()
    new_metrics = _parse_mean_line(output)
    print(f"    R@1={new_metrics.r1:.3f} AR@5={new_metrics.ar5:.2f}")
    return new_metrics


def _decide_keep(
    new: _Metrics, prev_r1: float, cycle: int,
) -> tuple[Literal['kept', 'discarded'], float]:
    """Keep/discard decision. Returns (status, delta)."""
    delta = new.r1 - prev_r1
    improved = new.r1 > prev_r1
    ar5_ok = new.ar5 >= _AR5_KEEP_THRESHOLD
    if improved and ar5_ok:
        print(f"  ✓ KEEP — R@1 {prev_r1:.3f}→{new.r1:.3f}")
        return 'kept', delta
    reason = "AR@5 dropped" if not ar5_ok else "R@1 did not improve"
    print(f"  ✗ DISCARD — {reason} (delta={delta:+.3f})")
    _log_dead_end(cycle, new.r1, delta, reason)
    _restore_model_backup()
    return 'discarded', delta


def _restore_model_backup() -> None:
    backup = _MODEL_DIR.with_suffix('.bak')
    if backup.exists():
        shutil.rmtree(_MODEL_DIR)
        backup.rename(_MODEL_DIR)


def _backup_model() -> None:
    backup = _MODEL_DIR.with_suffix('.bak')
    if _MODEL_DIR.exists():
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(_MODEL_DIR, backup)


def _cleanup_backup() -> None:
    backup = _MODEL_DIR.with_suffix('.bak')
    if backup.exists():
        shutil.rmtree(backup)


def run_cycle(cycle_num: int, prev_r1: float) -> CycleResult:
    """Execute one self-improvement cycle."""
    print(f"\n{'=' * 60}")
    print(f"  CYCLE {cycle_num}")
    print(f"{'=' * 60}")
    _metrics, hard_neg, training_size = _eval_and_mine(cycle_num)
    new_metrics = _retrain_and_reeval()
    status, delta = _decide_keep(new_metrics, prev_r1, cycle_num)
    result = CycleResult(
        cycle_num=cycle_num, r1=new_metrics.r1, r3=new_metrics.r3,
        mrr=new_metrics.mrr, ar5=new_metrics.ar5,
        latency_s=new_metrics.latency_s, status=status,
        hard_neg_count=hard_neg, training_size=training_size,
    )
    desc = f"autorecall cycle {cycle_num} (delta={delta:+.3f})"
    _log_result(result, desc)
    _log_convergence(cycle_num, new_metrics.r1, training_size, hard_neg)
    return result


def _execute_cycle(
    cycle: int, current_r1: float, r1_history: list[float],
) -> float:
    """Run one cycle, update history, return new current_r1."""
    _backup_model()
    result = run_cycle(cycle, current_r1)
    new_r1 = result.r1 if result.status == 'kept' else current_r1
    r1_history.append(new_r1)
    _cleanup_backup()
    return new_r1


def _run_loop(max_cycles: int, threshold: float) -> None:
    """Main loop: baseline → N cycles → convergence check."""
    print("\nEstablishing baseline...")
    baseline = _parse_mean_line(_run_eval())
    print(f"Baseline: R@1={baseline.r1:.3f} AR@5={baseline.ar5:.2f}")
    r1_history: list[float] = [baseline.r1]
    current_r1 = baseline.r1
    for cycle in range(1, max_cycles + 1):
        current_r1 = _execute_cycle(cycle, current_r1, r1_history)
        if _check_convergence(r1_history, threshold):
            print(f"\n✓ Converged after {cycle} cycles")
            break
    print(f"\nFinal: R@1={current_r1:.3f} (delta={current_r1 - baseline.r1:+.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoRecall")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--convergence-threshold", type=float, default=_DEFAULT_THRESHOLD)
    args = parser.parse_args()
    print("AutoRecall — Self-Improving Eval Loop")
    _run_loop(args.cycles, args.convergence_threshold)


if __name__ == "__main__":
    main()
