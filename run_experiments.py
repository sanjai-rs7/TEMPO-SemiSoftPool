"""
TEMPO Experiment Runner
=======================
Run all relevant training/evaluation combinations from the paper.
Usage on Kaggle: Just run this cell after setup.
Usage locally:   python run_experiments.py
"""

import subprocess
import os
import json
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

BASE_CMD = [
    "python", "train_TEMPO.py",
    "--config_path", "./configs/etth1_local.yml",
    "--model", "TEMPO",
    "--seq_len", "336",
    "--pred_len", "96",
    "--batch_size", "64",
    "--train_epochs", "10",
    "--gpt_layers", "6",
    "--d_model", "768",
    "--patch_size", "16",
    "--stride", "8",
    "--prompt", "1",
    "--pretrain", "1",
    "--freeze", "1",
    "--is_gpt", "1",
    "--num_nodes", "1",
    "--loss_func", "mse",
    "--stl_weight", "0.01",
    "--learning_rate", "0.001",
    "--checkpoints", "./checkpoints/",
    "--itr", "1",
    "--equal", "1",
    "--use_token", "0",
]

# ============================================
# ALL 10 EXPERIMENTS
# ============================================

experiments = [

    # ------------------------------------------------------------------
    # EXPERIMENT 1: Single dataset train & test (baseline sanity check)
    # Train on ETTh1, test on ETTh1
    # ------------------------------------------------------------------
    {
        "name": "1_single_ETTh1",
        "desc": "Single dataset: Train ETTh1 → Test ETTh1",
        "datasets": "ETTh1",
        "eval_data": "ETTh1",
        "target_data": "ETTh1",
        "pool": False,
        "semi_soft_pool": False,
    },

    # ------------------------------------------------------------------
    # EXPERIMENT 2: Single dataset with Semi-Soft Pool (your novelty)
    # Train on ETTh1, test on ETTh1, using semi-soft pool
    # ------------------------------------------------------------------
    {
        "name": "2_single_ETTh1_ssp",
        "desc": "Single dataset + Semi-Soft Pool: Train ETTh1 → Test ETTh1",
        "datasets": "ETTh1",
        "eval_data": "ETTh1",
        "target_data": "ETTh1",
        "pool": True,
        "semi_soft_pool": True,
    },

    # ------------------------------------------------------------------
    # EXPERIMENT 3: Many-to-one zero-shot (paper's main setting)
    # Train on 6 datasets → Test on Weather (unseen)
    # This is the EXACT setting from Table 1 of the paper
    # ------------------------------------------------------------------
    {
        "name": "3_zero_shot_weather",
        "desc": "Zero-shot (paper): Train 6 datasets → Test Weather (unseen)",
        "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,electricity,traffic",
        "eval_data": "ETTm1",
        "target_data": "weather",
        "pool": False,
        "semi_soft_pool": False,
    },

    # ------------------------------------------------------------------
    # EXPERIMENT 4: Same as Exp 3 WITH Semi-Soft Pool (YOUR NOVELTY)
    # ------------------------------------------------------------------
    {
        "name": "4_zero_shot_weather_ssp",
        "desc": "Zero-shot + Semi-Soft Pool: Train 6 → Test Weather (unseen)",
        "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,electricity,traffic",
        "eval_data": "ETTm1",
        "target_data": "weather",
        "pool": True,
        "semi_soft_pool": True,
    },

    # ------------------------------------------------------------------
    # EXPERIMENT 5: Zero-shot on ETTm1 (unseen)
    # Train on 6 datasets (excluding ETTm1) → Test on ETTm1
    # ------------------------------------------------------------------
    {
        "name": "5_zero_shot_ETTm1",
        "desc": "Zero-shot: Train 6 datasets → Test ETTm1 (unseen)",
        "datasets": "ETTm2,ETTh1,ETTh2,electricity,traffic,weather",
        "eval_data": "ETTh1",
        "target_data": "ETTm1",
        "pool": False,
        "semi_soft_pool": False,
    },

    # ------------------------------------------------------------------
    # EXPERIMENT 6: Same as Exp 5 WITH Semi-Soft Pool
    # ------------------------------------------------------------------
    {
        "name": "6_zero_shot_ETTm1_ssp",
        "desc": "Zero-shot + Semi-Soft Pool: Train 6 → Test ETTm1 (unseen)",
        "datasets": "ETTm2,ETTh1,ETTh2,electricity,traffic,weather",
        "eval_data": "ETTh1",
        "target_data": "ETTm1",
        "pool": True,
        "semi_soft_pool": True,
    },

    # ------------------------------------------------------------------
    # EXPERIMENT 7: Zero-shot on Electricity (unseen)
    # Train on 6 datasets (excluding electricity) → Test on Electricity
    # ------------------------------------------------------------------
    {
        "name": "7_zero_shot_electricity",
        "desc": "Zero-shot: Train 6 datasets → Test Electricity (unseen)",
        "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,traffic,weather",
        "eval_data": "ETTm1",
        "target_data": "electricity",
        "pool": False,
        "semi_soft_pool": False,
    },

    # ------------------------------------------------------------------
    # EXPERIMENT 8: Same as Exp 7 WITH Semi-Soft Pool
    # ------------------------------------------------------------------
    {
        "name": "8_zero_shot_electricity_ssp",
        "desc": "Zero-shot + Semi-Soft Pool: Train 6 → Test Electricity (unseen)",
        "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,traffic,weather",
        "eval_data": "ETTm1",
        "target_data": "electricity",
        "pool": True,
        "semi_soft_pool": True,
    },

    # ------------------------------------------------------------------
    # EXPERIMENT 9: Random Pool baseline (paper's pool, random init)
    # Same training as Exp 3 but with random pool (to compare against your novelty)
    # ------------------------------------------------------------------
    {
        "name": "9_zero_shot_weather_random_pool",
        "desc": "Zero-shot + Random Pool (paper): Train 6 → Test Weather",
        "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,electricity,traffic",
        "eval_data": "ETTm1",
        "target_data": "weather",
        "pool": True,
        "semi_soft_pool": False,
    },

    # ------------------------------------------------------------------
    # EXPERIMENT 10: No prompt at all (ablation)
    # Same training as Exp 3 but prompt=0
    # ------------------------------------------------------------------
    {
        "name": "10_zero_shot_weather_no_prompt",
        "desc": "Ablation: Zero-shot WITHOUT any prompt → Test Weather",
        "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,electricity,traffic",
        "eval_data": "ETTm1",
        "target_data": "weather",
        "pool": False,
        "semi_soft_pool": False,
        "no_prompt": True,
    },
]


# ============================================
# RUNNER
# ============================================

def run_experiment(exp, results_file="experiment_results.json"):
    """Run a single experiment and log results."""
    print("\n" + "=" * 60)
    print(f"  EXPERIMENT: {exp['name']}")
    print(f"  {exp['desc']}")
    print("=" * 60)

    cmd = BASE_CMD.copy()
    cmd += ["--datasets", exp["datasets"]]
    cmd += ["--eval_data", exp["eval_data"]]
    cmd += ["--target_data", exp["target_data"]]
    cmd += ["--model_id", exp["name"]]

    if exp.get("pool"):
        cmd += ["--pool"]
    if exp.get("semi_soft_pool"):
        cmd += ["--semi_soft_pool"]
    if exp.get("no_prompt"):
        cmd += ["--prompt", "0"]
        # Remove the --prompt 1 from BASE_CMD
        idx = cmd.index("--prompt")
        cmd[idx + 1] = "0"

    print(f"  Command: {' '.join(cmd)}\n")

    start = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (datetime.now() - start).total_seconds()

    # Print output
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"  ❌ FAILED: {result.stderr[-500:]}")
        return None

    # Parse MSE/MAE from output
    mse, mae = None, None
    for line in result.stdout.split("\n"):
        if "mse:" in line and "mae:" in line:
            parts = line.split("mse:")[1]
            mse = float(parts.split(" ")[0])
            mae = float(parts.split("mae:")[1].strip())

    entry = {
        "name": exp["name"],
        "desc": exp["desc"],
        "datasets": exp["datasets"],
        "target": exp["target_data"],
        "pool": exp.get("pool", False),
        "semi_soft_pool": exp.get("semi_soft_pool", False),
        "mse": mse,
        "mae": mae,
        "time_seconds": round(elapsed),
    }

    # Append to results file
    results = []
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
    results.append(entry)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✅ MSE: {mse}  MAE: {mae}  Time: {elapsed:.0f}s")
    return entry


def print_results_table(results_file="experiment_results.json"):
    """Print a nice comparison table."""
    if not os.path.exists(results_file):
        print("No results yet.")
        return

    with open(results_file) as f:
        results = json.load(f)

    print("\n" + "=" * 90)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'#':<4} {'Name':<35} {'Target':<8} {'MSE':<8} {'MAE':<8} {'Time':<8} {'Pool':<6}")
    print("-" * 90)
    for r in results:
        pool_str = "SSP" if r.get("semi_soft_pool") else ("RP" if r.get("pool") else "—")
        mse_str = f"{r['mse']:.4f}" if r['mse'] else "FAIL"
        mae_str = f"{r['mae']:.4f}" if r['mae'] else "FAIL"
        time_str = f"{r['time_seconds']}s"
        print(f"{r['name'][:3]:<4} {r['desc'][:35]:<35} {r['target']:<8} {mse_str:<8} {mae_str:<8} {time_str:<8} {pool_str:<6}")

    # Key comparisons
    print("\n" + "-" * 90)
    print("  KEY COMPARISONS:")
    by_name = {r["name"]: r for r in results}

    pairs = [
        ("3_zero_shot_ETTh1", "4_zero_shot_ETTh1_ssp", "Semi-Soft Pool vs Semi-Soft (ETTh1 zero-shot)"),
        ("5_zero_shot_ETTm1", "6_zero_shot_ETTm1_ssp", "Semi-Soft Pool vs Semi-Soft (ETTm1 zero-shot)"),
        ("9_zero_shot_ETTh1_random_pool", "4_zero_shot_ETTh1_ssp", "Semi-Soft Pool vs Random Pool"),
        ("10_zero_shot_ETTh1_no_prompt", "3_zero_shot_ETTh1", "With Prompt vs Without Prompt"),
    ]
    for base, novel, desc in pairs:
        if base in by_name and novel in by_name:
            b, n = by_name[base], by_name[novel]
            if b["mse"] and n["mse"]:
                diff = ((b["mse"] - n["mse"]) / b["mse"]) * 100
                winner = "✅ NOVELTY WINS" if diff > 0 else "❌ Baseline wins"
                print(f"  {desc}: {diff:+.1f}% MSE change → {winner}")


# ============================================
# MAIN — Choose which experiments to run
# ============================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all",
                        help="Which experiments: 'all', '1,2,3', 'quick', 'novelty'")
    args = parser.parse_args()

    if args.exp == "all":
        to_run = experiments
    elif args.exp == "quick":
        # Just the key comparison: semi-soft vs semi-soft pool on ETTh1
        to_run = [experiments[0], experiments[1]]
    elif args.exp == "novelty":
        # All novelty comparisons: 3 vs 4, 5 vs 6, 9 vs 4
        to_run = [experiments[2], experiments[3], experiments[4], experiments[5], experiments[8]]
    else:
        indices = [int(x) - 1 for x in args.exp.split(",")]
        to_run = [experiments[i] for i in indices]

    print(f"\n🚀 Running {len(to_run)} experiments...\n")
    for exp in to_run:
        run_experiment(exp)

    print_results_table()
