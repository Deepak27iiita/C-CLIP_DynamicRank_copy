"""
compare_bandit_results.py
--------------------------
Reads the JSON files produced by each bandit run and prints a clean
comparison table:

  - Per-task rank chosen by each algorithm
  - Per-task reward received by each algorithm
  - Final mean reward per algorithm
  - Best rank selected per algorithm

Usage (from repo root, after training is done):
    python scripts/compare_bandit_results.py

Reads from:
    checkpoints/comparison/ucb1/bandit_state.json
    checkpoints/comparison/epsilon_greedy/bandit_state.json
    checkpoints/comparison/thompson/bandit_state.json
"""

import os
import json

# ── Where each algorithm saves its bandit state ───────────────────────────────
ALGO_PATHS = {
    "UCB1":           "checkpoints/comparison/ucb1/bandit_state.json",
    "ε-Greedy":       "checkpoints/comparison/epsilon_greedy/bandit_state.json",
    "Thompson":       "checkpoints/comparison/thompson/bandit_state.json",
}

TASK_NAMES = ["fgvc_aircraft", "dtd", "eurosat", "flowers102", "oxford_pets"]


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def build_task_table(states: dict) -> None:
    """Print per-task rank and reward for each algorithm."""
    print("\n" + "="*75)
    print(" PER-TASK RANK SELECTION")
    print("="*75)
    header = f"{'Task':<18}" + "".join(f"{'Algo':>16}" for algo in ALGO_PATHS)
    print(f"{'Task':<18}" + "".join(f"{algo:>16}" for algo in ALGO_PATHS))
    print("-"*75)

    for task_idx, task_name in enumerate(TASK_NAMES):
        row = f"{task_name:<18}"
        for algo, state in states.items():
            history = state.get("task_history", [])
            entry = next((h for h in history if h.get("task_idx") == task_idx), None)
            if entry:
                row += f"  r={entry['rank_chosen']:>2}            "
            else:
                row += f"  {'N/A':>14}  "
        print(row)

    print("\n" + "="*75)
    print(" PER-TASK REWARD")
    print("="*75)
    print(f"{'Task':<18}" + "".join(f"{algo:>16}" for algo in ALGO_PATHS))
    print("-"*75)

    for task_idx, task_name in enumerate(TASK_NAMES):
        row = f"{task_name:<18}"
        for algo, state in states.items():
            history = state.get("task_history", [])
            entry = next((h for h in history if h.get("task_idx") == task_idx), None)
            if entry:
                row += f"  {entry['reward']:>8.4f}      "
            else:
                row += f"  {'N/A':>14}  "
        print(row)


def build_summary_table(states: dict) -> None:
    """Print overall summary per algorithm."""
    print("\n" + "="*75)
    print(" SUMMARY — OVERALL BANDIT PERFORMANCE")
    print("="*75)
    print(f"{'Metric':<28}" + "".join(f"{algo:>16}" for algo in ALGO_PATHS))
    print("-"*75)

    # Mean reward across all tasks
    row_mean = f"{'Mean reward':<28}"
    for algo, state in states.items():
        history = state.get("task_history", [])
        if history:
            mean_r = sum(h["reward"] for h in history) / len(history)
            row_mean += f"  {mean_r:>8.4f}      "
        else:
            row_mean += f"  {'N/A':>14}  "
    print(row_mean)

    # Best rank chosen
    row_best = f"{'Best rank (by mean reward)':<28}"
    for algo, state in states.items():
        best = state.get("best_rank", "N/A")
        row_best += f"  r={best!s:<13}  "
    print(row_best)

    # Total tasks seen
    row_tasks = f"{'Tasks completed':<28}"
    for algo, state in states.items():
        total = state.get("total_tasks", "N/A")
        row_tasks += f"  {total!s:>14}  "
    print(row_tasks)

    print("="*75)


def build_arm_table(states: dict) -> None:
    """Print per-rank arm statistics."""
    print("\n" + "="*75)
    print(" ARM STATISTICS (mean reward per rank)")
    print("="*75)
    print(f"{'Rank':<10}" + "".join(f"{algo:>21}" for algo in ALGO_PATHS))
    print("-"*75)

    rank_choices = [4, 8, 16, 32]
    for r in rank_choices:
        row = f"r={r:<8}"
        for algo, state in states.items():
            arms = state.get("arms", {})
            arm = arms.get(str(r), arms.get(r, {}))
            if arm:
                pulls = arm.get("n_pulls", 0)
                mean  = arm.get("mean_reward", 0.0)
                row += f"  pulls={pulls} mean={mean:.3f}   "
            else:
                row += f"  {'N/A':>18}  "
        print(row)

    print("="*75)


def main():
    states = {}
    for algo, path in ALGO_PATHS.items():
        state = load_state(path)
        if not state:
            print(f"[WARNING] No data for {algo} at {path}")
        states[algo] = state

    if all(not s for s in states.values()):
        print("\n[ERROR] No results found. Run training first:")
        print("  python run_bandit_comparison.py\n")
        return

    print("\n" + "="*75)
    print("  C-CLIP BANDIT ALGORITHM COMPARISON")
    print("  UCB1  vs  ε-Greedy  vs  Thompson Sampling")
    print("="*75)

    build_task_table(states)
    build_arm_table(states)
    build_summary_table(states)

    print("\nWINNER (by mean reward across 5 tasks):")
    best_algo = None
    best_mean = -1.0
    for algo, state in states.items():
        history = state.get("task_history", [])
        if history:
            mean_r = sum(h["reward"] for h in history) / len(history)
            if mean_r > best_mean:
                best_mean = mean_r
                best_algo = algo
    if best_algo:
        print(f"  ➤  {best_algo}  (mean reward = {best_mean:.4f})\n")
    else:
        print("  No complete results yet.\n")


if __name__ == "__main__":
    main()
