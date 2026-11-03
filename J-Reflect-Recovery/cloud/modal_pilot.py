"""Run the J-Reflect Recovery pilot unattended on Modal, under a dollar budget.

Usage, from the J-Reflect-Recovery directory:

    modal run --detach cloud/modal_pilot.py                 # whole pilot: dev -> freeze -> test
    modal run --detach cloud/modal_pilot.py --stage dev     # development split only, then stop
    modal run cloud/modal_pilot.py --stage status           # spend so far (no GPU)

--budget sets the total spend for this run in dollars (default 20, never above HARD_CAP).
Spend is tracked across invocations in the volume. When the budget runs out, the current
command is stopped, progress is saved, and a later invocation with more budget resumes.
Before the test split starts, its cost is projected from the development split; if the
projection does not fit in the remaining budget, the job stops instead of leaving an
unfinished test split.

With --detach the job keeps running after this computer sleeps or disconnects.
Results: modal volume get jrecovery-runs runs/pilot runs/modal-pilot
"""

import json
import os
import signal
import subprocess
import threading
import time
import tomllib
from pathlib import Path

import modal

# Billing: Modal pricing page, November 2026 (per second).
GPU = "L40S"
CPU_CORES = 2
MEMORY_MIB = 32768
RATE_PER_SECOND = 0.000542 + CPU_CORES * 0.0000131 + (MEMORY_MIB / 1024) * 0.00000222  # ~$2.30/h
STARTUP_SURCHARGE_S = 120  # container start and image load, billed before this code runs
HARD_CAP = 20.0  # dollars, total across all invocations
DEFAULT_BUDGET = 20.0

PROJECT = Path(__file__).resolve().parents[1]
WORK = "/work/runs"  # container-local disk: fast, supports file locks
SAVED = "/vol/runs"  # persistent Modal Volume
RUN = f"{WORK}/pilot"
LEDGER = f"{WORK}/budget.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("rsync")
    .pip_install("uv")
    .env({"UV_LINK_MODE": "copy"})
    .add_local_dir(
        PROJECT,
        "/app",
        copy=True,
        ignore=[
            ".venv",
            ".cache",
            "runs",
            "colab",
            "cloud",
            "**/__pycache__",
            ".pytest_cache",
            ".ruff_cache",
        ],
    )
    # Install exactly the versions in uv.lock. --frozen skips re-resolving, which can differ
    # across uv versions; the lock was checked against pyproject.toml locally (uv lock --check).
    # The run records Python/package versions; they stay fixed while this image is cached.
    .run_commands(
        "uv --version",
        "cd /app && uv sync --frozen --extra dev --extra model --managed-python --python 3.12",
    )
)

app = modal.App("jrecovery-pilot", image=image)
runs_volume = modal.Volume.from_name("jrecovery-runs", create_if_missing=True)
sync_lock = threading.Lock()


class BudgetReached(Exception):
    pass


class Meter:
    """Cumulative billed-time estimate, persisted in the volume."""

    def __init__(self, budget):
        ledger = json.loads(Path(LEDGER).read_text()) if Path(LEDGER).exists() else {}
        self.previous = ledger.get("spent_seconds", 0.0) + STARTUP_SURCHARGE_S
        self.invocations = ledger.get("invocations", 0) + 1
        self.started = time.time()
        self.budget = budget

    def spent(self):
        return (self.previous + time.time() - self.started) * RATE_PER_SECOND

    def remaining_seconds(self):
        return (self.budget - self.spent()) / RATE_PER_SECOND

    def write(self):
        Path(LEDGER).write_text(
            json.dumps(
                {
                    "spent_seconds": self.previous + time.time() - self.started,
                    "spent_dollars_estimate": round(self.spent(), 2),
                    "invocations": self.invocations,
                    "rate_per_hour": round(RATE_PER_SECOND * 3600, 3),
                    "gpu": GPU,
                },
                indent=2,
            )
        )

    def line(self):
        return f"spent ~${self.spent():.2f} of ${self.budget:.2f} budget"


def save(meter=None):
    """Write the ledger, copy the local run to the volume, and commit it."""
    with sync_lock:
        if meter is not None:
            meter.write()
        subprocess.run(
            ["rsync", "-a", "--exclude", "*.tmp", "--exclude", ".lock", f"{WORK}/", f"{SAVED}/"],
            check=True,
        )
        runs_volume.commit()


def autosave(meter, stop, every=300):
    while not stop.wait(every):
        try:
            save(meter)
            print(f"[autosave] {meter.line()}", flush=True)
        except Exception as exc:  # keep running; the next save retries
            print(f"[autosave] failed: {exc}", flush=True)


def sh(cmd, meter):
    """Run a command, killing it if the budget runs out."""
    remaining = meter.remaining_seconds() - 60  # leave time to save
    if remaining <= 0:
        raise BudgetReached(meter.line())
    print(f"\n$ {cmd}   [{meter.line()}]", flush=True)
    process = subprocess.Popen(["bash", "-c", cmd], start_new_session=True)
    try:
        code = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        raise BudgetReached(meter.line()) from None
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def jr(meter, *args):
    try:
        sh("cd /app && .venv/bin/jrecovery " + " ".join(args), meter)
    finally:
        save(meter)


def compute_seconds(split):
    """Generation and readout seconds recorded in a split's saved records."""
    total = 0.0
    for stage in ["first", "correction", "ablation"]:
        for path in Path(RUN, stage, split).rglob("*.json"):
            payload = json.loads(path.read_text()).get("payload", {})
            total += payload.get("generation", {}).get("seconds", 0.0)
            total += payload.get("readout_seconds", 0.0)
    return total


def projected_test_dollars():
    data = tomllib.loads(Path("/app/configs/pilot.toml").read_text())["data"]
    ratio = data["test_count"] / data["dev_count"]
    # Recorded compute understates wall time (model loads, I/O, Python); pad it.
    seconds = compute_seconds("dev") * ratio * 1.3 + 4 * 180
    return seconds * RATE_PER_SECOND


def check_test_fits(meter, force):
    if Path(RUN, "first", "test").exists():
        return  # test already started; finishing it is the best use of remaining budget
    projected = projected_test_dollars()
    available = meter.budget - meter.spent()
    print(f"Projected test-split cost ~${projected:.2f}; ${available:.2f} left.", flush=True)
    if projected > available and not force:
        raise BudgetReached(
            f"test split projected at ~${projected:.2f} but only ${available:.2f} remains. "
            f"Rerun with a larger --budget (max ${HARD_CAP:.0f}) or --force to start anyway."
        )


@app.function(
    gpu=GPU,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    volumes={"/vol": runs_volume},
    timeout=int(HARD_CAP / RATE_PER_SECOND),  # absolute backstop for one invocation
)
def pilot(stage: str = "all", budget: float = DEFAULT_BUDGET, force: bool = False):
    if stage not in {"dev", "all"}:
        raise ValueError("stage must be 'dev' or 'all'")
    budget = min(budget, HARD_CAP)
    Path(WORK).mkdir(parents=True, exist_ok=True)
    Path(SAVED).mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{SAVED}/", f"{WORK}/"], check=True)  # resume
    meter = Meter(budget)
    print(f"Starting on {GPU} at ~${RATE_PER_SECOND * 3600:.2f}/h; {meter.line()}", flush=True)

    stop = threading.Event()
    threading.Thread(target=autosave, args=(meter, stop), daemon=True).start()
    try:
        sh("cd /app && .venv/bin/jrecovery doctor", meter)
        if not Path(f"{RUN}/run.json").exists():
            jr(meter, "prepare", "--config", "configs/pilot.toml", "--run", RUN)
        jr(meter, "collect", "--run", RUN, "--split", "dev")
        jr(meter, "correct", "--run", RUN, "--split", "dev")
        jr(meter, "ablate", "--run", RUN, "--split", "dev")
        jr(meter, "report", "--run", RUN, "--split", "dev")
        print(Path(f"{RUN}/reports/dev/report.md").read_text(), flush=True)
        print(f"Projected test-split cost ~${projected_test_dollars():.2f}", flush=True)
        if stage == "dev":
            return

        if not Path(f"{RUN}/frozen.json").exists():
            jr(meter, "freeze", "--run", RUN)  # fails visibly if the development gate fails
        check_test_fits(meter, force)
        jr(meter, "collect", "--run", RUN, "--split", "test")
        jr(meter, "correct", "--run", RUN, "--split", "test")
        jr(meter, "ablate", "--run", RUN, "--split", "test")
        jr(meter, "report", "--run", RUN, "--split", "test")
        print(Path(f"{RUN}/reports/test/report.md").read_text(), flush=True)
    except BudgetReached as exc:
        print(f"\nSTOPPED TO STAY IN BUDGET: {exc}. Progress is saved.", flush=True)
    finally:
        stop.set()
        save(meter)
        print(f"Done for now: {meter.line()} (estimate; Modal's billing page is authoritative)")


@app.function(volumes={"/vol": runs_volume}, cpu=0.25)
def status():
    path = Path(SAVED, "budget.json")
    print(path.read_text() if path.exists() else "No spend recorded yet.")


@app.local_entrypoint()
def main(stage: str = "all", budget: float = DEFAULT_BUDGET, force: bool = False):
    if stage == "status":
        status.remote()
    else:
        pilot.remote(stage, budget, force)
