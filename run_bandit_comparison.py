"""
run_bandit_comparison.py
------------------------
Runs all three bandit algorithms (UCB1, Epsilon-Greedy, Thompson Sampling)
sequentially on the same 5-task continual learning benchmark and saves
combined results to checkpoints/comparison/comparison_results.json

Usage (from repo root):
    python run_bandit_comparison.py

Each algorithm gets a fresh model (same pretrained CLIP weights) so results
are directly comparable. All checkpoints and bandit logs are stored in
separate subdirectories under checkpoints/comparison/.
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime

# ── Config paths for each algorithm ──────────────────────────────────────────
CONFIGS = {
    "ucb1":            "configs/bandit_ucb1.yaml",
    "epsilon_greedy":  "configs/bandit_epsilon_greedy.yaml",
    "thompson":        "configs/bandit_thompson.yaml",
}

RESULTS_DIR  = "checkpoints/comparison"
RESULTS_FILE = os.path.join(RESULTS_DIR, "comparison_results.json")

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_one_algorithm(algo_name: str, config_path: str) -> dict:
    """
    Runs train_bandit.py for one algorithm and returns its bandit history.
    """
    print(f"\n{'='*65}")
    print(f"  RUNNING: {algo_name.upper()}")
    print(f"  Config : {config_path}")
    print(f"{'='*65}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "train_bandit.py", "--config", config_path],
        check=True,   # raises CalledProcessError if training crashes
    )
    elapsed = time.time() - start
    print(f"\n[✓] {algo_name} finished in {elapsed/60:.1f} min")

    # Load the saved bandit state for this algorithm
    bandit_state_path = os.path.join(
        RESULTS_DIR, algo_name, "bandit_state.json"
    )
    if os.path.exists(bandit_state_path):
        with open(bandit_state_path) as f:
            bandit_data = json.load(f)
    else:
        bandit_data = {}
        print(f"[WARNING] Bandit state not found at {bandit_state_path}")

    return {
        "algorithm":      algo_name,
        "elapsed_min":    round(elapsed / 60, 2),
        "bandit_history": bandit_data,
    }


def main():
    all_results = {
        "timestamp":  datetime.now().isoformat(),
        "algorithms": {},
    }

    for algo_name, config_path in CONFIGS.items():
        if not os.path.exists(config_path):
            print(f"[SKIP] Config not found: {config_path}")
            continue

        try:
            result = run_one_algorithm(algo_name, config_path)
            all_results["algorithms"][algo_name] = result
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {algo_name} training failed: {e}")
            all_results["algorithms"][algo_name] = {
                "algorithm": algo_name,
                "error": str(e),
            }

        # Save incrementally after each run so a crash doesn't lose data
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[Saved] Intermediate results → {RESULTS_FILE}")

    print(f"\n{'='*65}")
    print("  ALL RUNS COMPLETE")
    print(f"  Results saved → {RESULTS_FILE}")
    print(f"{'='*65}")
    print("\nRun this next to see the comparison table:")
    print("  python scripts/compare_bandit_results.py\n")


if __name__ == "__main__":
    main()
