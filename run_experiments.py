"""
TEMPO Experiment Runner v2
==========================
Full experiment pipeline with logging, plots, tables, and model saving.
Usage:
  python run_experiments.py --exp quick       # Fast test (2 experiments)
  python run_experiments.py --exp novelty     # Key thesis results (5 experiments)
  python run_experiments.py --exp all         # All 10 experiments
  python run_experiments.py --exp 3,4,9       # Specific experiments
"""

import subprocess, os, json, time, sys
import numpy as np
from datetime import datetime

RESULTS_DIR = "./experiment_results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.json")

BASE_ARGS = {
    "config_path": "./configs/etth1_local.yml",
    "model": "TEMPO",
    "seq_len": "336",
    "pred_len": "96",
    "batch_size": "64",
    "train_epochs": "10",
    "gpt_layers": "6",
    "d_model": "768",
    "patch_size": "16",
    "stride": "8",
    "prompt": "1",
    "pretrain": "1",
    "freeze": "1",
    "is_gpt": "1",
    "num_nodes": "1",
    "loss_func": "mse",
    "stl_weight": "0.01",
    "learning_rate": "0.001",
    "checkpoints": "./checkpoints/",
    "itr": "1",
    "equal": "1",
    "use_token": "0",
    "percent": "100",
}

experiments = [
    {"name": "1_single_ETTh1",                "desc": "Single: ETTh1 → ETTh1 (semi-soft)",                    "datasets": "ETTh1",                                    "eval_data": "ETTh1", "target_data": "ETTh1",       "pool": False, "semi_soft_pool": False},
    {"name": "2_single_ETTh1_ssp",            "desc": "Single + Semi-Soft Pool: ETTh1 → ETTh1",              "datasets": "ETTh1",                                    "eval_data": "ETTh1", "target_data": "ETTh1",       "pool": True,  "semi_soft_pool": True},
    {"name": "3_zero_shot_weather",            "desc": "Zero-shot: 6 datasets → Weather (semi-soft)",         "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,electricity,traffic", "eval_data": "ETTm1", "target_data": "weather",  "pool": False, "semi_soft_pool": False},
    {"name": "4_zero_shot_weather_ssp",        "desc": "Zero-shot + SSP: 6 datasets → Weather",              "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,electricity,traffic", "eval_data": "ETTm1", "target_data": "weather",  "pool": True,  "semi_soft_pool": True},
    {"name": "5_zero_shot_ETTm1",              "desc": "Zero-shot: 6 datasets → ETTm1 (semi-soft)",          "datasets": "ETTm2,ETTh1,ETTh2,electricity,traffic,weather","eval_data": "ETTh1", "target_data": "ETTm1",   "pool": False, "semi_soft_pool": False},
    {"name": "6_zero_shot_ETTm1_ssp",          "desc": "Zero-shot + SSP: 6 datasets → ETTm1",                "datasets": "ETTm2,ETTh1,ETTh2,electricity,traffic,weather","eval_data": "ETTh1", "target_data": "ETTm1",   "pool": True,  "semi_soft_pool": True},
    {"name": "7_zero_shot_electricity",         "desc": "Zero-shot: 6 datasets → Electricity (semi-soft)",    "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,traffic,weather",     "eval_data": "ETTm1", "target_data": "electricity","pool": False, "semi_soft_pool": False},
    {"name": "8_zero_shot_electricity_ssp",     "desc": "Zero-shot + SSP: 6 datasets → Electricity",          "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,traffic,weather",     "eval_data": "ETTm1", "target_data": "electricity","pool": True,  "semi_soft_pool": True},
    {"name": "9_zero_shot_weather_random_pool", "desc": "Zero-shot + Random Pool: 6 datasets → Weather",      "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,electricity,traffic", "eval_data": "ETTm1", "target_data": "weather",  "pool": True,  "semi_soft_pool": False},
    {"name": "10_zero_shot_weather_no_prompt",  "desc": "Ablation: No prompt → Weather",                       "datasets": "ETTm1,ETTm2,ETTh1,ETTh2,electricity,traffic", "eval_data": "ETTm1", "target_data": "weather",  "pool": False, "semi_soft_pool": False, "no_prompt": True},
]

def build_cmd(exp):
    cmd = ["python", "train_TEMPO.py"]
    args = BASE_ARGS.copy()
    args["datasets"] = exp["datasets"]
    args["eval_data"] = exp["eval_data"]
    args["target_data"] = exp["target_data"]
    args["model_id"] = exp["name"]
    if exp.get("no_prompt"):
        args["prompt"] = "0"
    for k, v in args.items():
        cmd += [f"--{k}", v]
    if exp.get("pool"):
        cmd += ["--pool"]
    if exp.get("semi_soft_pool"):
        cmd += ["--semi_soft_pool"]
    return cmd

def run_experiment(exp):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  EXPERIMENT: {exp['name']}")
    print(f"  {exp['desc']}")
    print(f"  Train: {exp['datasets']}")
    print(f"  Test:  {exp['target_data']}")
    prompt_type = "Semi-Soft Pool" if exp.get("semi_soft_pool") else "Random Pool" if exp.get("pool") else "No Prompt" if exp.get("no_prompt") else "Semi-Soft"
    print(f"  Prompt: {prompt_type}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(sep)

    cmd = build_cmd(exp)
    start = time.time()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    full_output = []
    for line in proc.stdout:
        full_output.append(line)
        # Show progress lines
        if any(k in line for k in ['iters:', 'Epoch:', 'Train Loss', 'Vali Loss', 'mse:', 'mae:', 'trainable params', 'Semi-Soft Pool', 'tqdm', '%|', 'speed:', 'EarlyStopping', 'Saving model']):
            print(f"  {line.rstrip()}")
    proc.wait()
    elapsed = time.time() - start
    output = "".join(full_output)

    # Parse MSE/MAE
    mse, mae = None, None
    for line in output.split("\n"):
        if "mse:" in line and "mae:" in line:
            try:
                parts = line.split("mse:")[1]
                mse = float(parts.split(" ")[0])
                mae = float(parts.split("mae:")[1].strip())
            except:
                pass

    entry = {
        "name": exp["name"],
        "desc": exp["desc"],
        "train_datasets": exp["datasets"],
        "test_dataset": exp["target_data"],
        "prompt_type": prompt_type,
        "mse": mse,
        "mae": mae,
        "time_seconds": round(elapsed),
        "time_human": f"{int(elapsed//60)}m {int(elapsed%60)}s",
        "timestamp": datetime.now().isoformat(),
        "success": proc.returncode == 0 and mse is not None,
    }

    # Save results
    results = load_results()
    # Replace if same experiment exists
    results = [r for r in results if r["name"] != exp["name"]]
    results.append(entry)
    save_results(results)

    # Save full log
    log_path = os.path.join(RESULTS_DIR, "logs", f"{exp['name']}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(output)

    status = "✅ SUCCESS" if entry["success"] else "❌ FAILED"
    print(f"\n  {status} | MSE: {mse} | MAE: {mae} | Time: {entry['time_human']}")
    return entry

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []

def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def print_results_table():
    results = load_results()
    if not results:
        print("No results yet.")
        return

    sep = "=" * 100
    print(f"\n{sep}")
    print("  EXPERIMENT RESULTS SUMMARY")
    print(sep)
    print(f"  {'#':<3} {'Description':<50} {'Target':<12} {'MSE':<10} {'MAE':<10} {'Prompt':<15} {'Time':<10}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["name"]):
        mse_str = f"{r['mse']:.4f}" if r.get('mse') else "FAIL"
        mae_str = f"{r['mae']:.4f}" if r.get('mae') else "FAIL"
        print(f"  {r['name'][:2]:<3} {r['desc'][:50]:<50} {r['test_dataset']:<12} {mse_str:<10} {mae_str:<10} {r['prompt_type']:<15} {r.get('time_human','?'):<10}")

    # Key comparisons
    by_name = {r["name"]: r for r in results}
    pairs = [
        ("3_zero_shot_weather", "4_zero_shot_weather_ssp", "Semi-Soft Pool vs Semi-Soft (Weather)"),
        ("5_zero_shot_ETTm1", "6_zero_shot_ETTm1_ssp", "Semi-Soft Pool vs Semi-Soft (ETTm1)"),
        ("7_zero_shot_electricity", "8_zero_shot_electricity_ssp", "Semi-Soft Pool vs Semi-Soft (Electricity)"),
        ("9_zero_shot_weather_random_pool", "4_zero_shot_weather_ssp", "Semi-Soft Pool vs Random Pool"),
        ("10_zero_shot_weather_no_prompt", "3_zero_shot_weather", "With Prompt vs Without"),
        ("1_single_ETTh1", "2_single_ETTh1_ssp", "Semi-Soft Pool vs Semi-Soft (Single)"),
    ]
    print(f"\n  KEY COMPARISONS:")
    print("-" * 100)
    for base_name, novel_name, desc in pairs:
        if base_name in by_name and novel_name in by_name:
            b, n = by_name[base_name], by_name[novel_name]
            if b.get("mse") and n.get("mse"):
                diff = ((b["mse"] - n["mse"]) / b["mse"]) * 100
                arrow = "↓" if diff > 0 else "↑"
                winner = "✅ NOVELTY WINS" if diff > 0 else "❌ Baseline wins"
                print(f"  {desc:<55} {arrow} {abs(diff):.1f}% MSE → {winner}")
    print(sep)

def generate_plots():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    results = load_results()
    if len(results) < 2:
        return

    plot_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: MSE comparison bar chart
    names = [r["name"].split("_", 1)[1][:25] for r in results if r.get("mse")]
    mses = [r["mse"] for r in results if r.get("mse")]
    maes = [r["mae"] for r in results if r.get("mae")]
    colors = ["#2ecc71" if r.get("semi_soft_pool") else "#e74c3c" if r.get("pool") else "#3498db" if not r.get("no_prompt") else "#95a5a6" for r in results if r.get("mse")]

    if names:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        bars1 = ax1.barh(names, mses, color=colors)
        ax1.set_xlabel("MSE (lower is better)")
        ax1.set_title("MSE Comparison Across Experiments")
        ax1.invert_yaxis()
        for bar, val in zip(bars1, mses):
            ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=8)

        bars2 = ax2.barh(names, maes, color=colors)
        ax2.set_xlabel("MAE (lower is better)")
        ax2.set_title("MAE Comparison Across Experiments")
        ax2.invert_yaxis()
        for bar, val in zip(bars2, maes):
            ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=8)

        # Legend
        from matplotlib.patches import Patch
        legend = [Patch(color="#2ecc71", label="Semi-Soft Pool (Novelty)"),
                  Patch(color="#e74c3c", label="Random Pool"),
                  Patch(color="#3498db", label="Semi-Soft"),
                  Patch(color="#95a5a6", label="No Prompt")]
        ax1.legend(handles=legend, loc="lower right", fontsize=8)
        plt.tight_layout()
        path = os.path.join(plot_dir, "mse_mae_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Saved: {path}")

    # Plot 2: Paired comparison (baseline vs novelty)
    pairs = [
        ("3_zero_shot_weather", "4_zero_shot_weather_ssp"),
        ("5_zero_shot_ETTm1", "6_zero_shot_ETTm1_ssp"),
        ("7_zero_shot_electricity", "8_zero_shot_electricity_ssp"),
        ("1_single_ETTh1", "2_single_ETTh1_ssp"),
    ]
    by_name = {r["name"]: r for r in results}
    pair_labels, base_vals, novel_vals = [], [], []
    for b, n in pairs:
        if b in by_name and n in by_name and by_name[b].get("mse") and by_name[n].get("mse"):
            pair_labels.append(by_name[b]["test_dataset"])
            base_vals.append(by_name[b]["mse"])
            novel_vals.append(by_name[n]["mse"])

    if pair_labels:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(pair_labels))
        w = 0.35
        ax.bar(x - w/2, base_vals, w, label="Semi-Soft (Baseline)", color="#3498db")
        ax.bar(x + w/2, novel_vals, w, label="Semi-Soft Pool (Yours)", color="#2ecc71")
        ax.set_ylabel("MSE (lower is better)")
        ax.set_title("Baseline vs Your Novelty — Zero-Shot MSE")
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels)
        ax.legend()
        for i, (b, n) in enumerate(zip(base_vals, novel_vals)):
            diff = ((b - n) / b) * 100
            color = "green" if diff > 0 else "red"
            ax.annotate(f"{diff:+.1f}%", xy=(i + w/2, n), ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(plot_dir, "baseline_vs_novelty.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Saved: {path}")

def list_prompts():
    print("\n" + "=" * 70)
    print("  SEMI-SOFT PROMPT POOL — 30 Semantic Templates")
    print("=" * 70)
    templates = [
        "Strong upward trend with consistent growth over time",
        "Gradual downward trend with slow decline in values",
        "Flat stable trend with no significant change",
        "Sudden sharp increase followed by plateau",
        "Accelerating exponential growth trend",
        "Decelerating trend approaching a ceiling",
        "V shaped recovery trend after a drop",
        "Step function trend with abrupt level shift",
        "Oscillating trend with long period fluctuation",
        "Linear trend with constant rate of change",
        "Strong daily periodic cycle repeating every day",
        "Weekly seasonal pattern with weekend effects",
        "Monthly recurring pattern with regular peaks",
        "High frequency oscillation with short period",
        "Low frequency seasonal cycle with long period",
        "Seasonal pattern with increasing amplitude over time",
        "Seasonal pattern with decreasing amplitude fading",
        "Double peak seasonal pattern within each cycle",
        "Asymmetric seasonal pattern with sharp rise slow fall",
        "Irregular seasonal pattern with varying cycle length",
        "Low noise residual with small random fluctuations",
        "High variance residual with large unpredictable spikes",
        "Residual with occasional outlier extreme values",
        "Clustered volatility residual with bursts of noise",
        "White noise residual with uniform random variation",
        "Residual with gradual variance increase over time",
        "Residual with autocorrelated sequential dependence",
        "Sparse residual with mostly zero and rare spikes",
        "Heavy tailed residual with frequent large deviations",
        "Residual showing regime change in noise level",
    ]
    categories = ["TREND"] * 10 + ["SEASONAL"] * 10 + ["RESIDUAL"] * 10
    for i, (cat, t) in enumerate(zip(categories, templates)):
        print(f"  [{i:2d}] [{cat:8s}] {t}")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TEMPO Experiment Runner")
    parser.add_argument("--exp", type=str, default="quick", help="quick | novelty | all | 1,2,3 | results | plots | prompts")
    parser.add_argument("--fast", action="store_true", help="Fast mode: 3 epochs, 10%% data, no pretrain")
    parser.add_argument("--medium", action="store_true", help="Medium mode: 5 epochs, pretrained GPT-2")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Apply speed modes
    if args.fast:
        BASE_ARGS["train_epochs"] = "3"
        BASE_ARGS["pretrain"] = "0"
        BASE_ARGS["batch_size"] = "32"
        BASE_ARGS["percent"] = "10"
        RESULTS_DIR = "./experiment_results_fast"
        RESULTS_FILE = os.path.join(RESULTS_DIR, "results.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print("⚡ FAST MODE: 3 epochs, 10% data, no pretrain (~3-5 min per experiment)")
    elif args.medium:
        BASE_ARGS["train_epochs"] = "5"
        BASE_ARGS["pretrain"] = "1"
        BASE_ARGS["batch_size"] = "64"
        RESULTS_DIR = "./experiment_results_medium"
        RESULTS_FILE = os.path.join(RESULTS_DIR, "results.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print("🔶 MEDIUM MODE: 5 epochs, pretrained GPT-2 (~15-20 min per experiment)")

    if args.exp == "results":
        print_results_table()
        sys.exit(0)
    elif args.exp == "plots":
        generate_plots()
        sys.exit(0)
    elif args.exp == "prompts":
        list_prompts()
        sys.exit(0)

    if args.exp == "all":
        to_run = experiments
    elif args.exp == "quick":
        to_run = [experiments[0], experiments[1]]
    elif args.exp == "novelty":
        to_run = [experiments[2], experiments[3], experiments[4], experiments[5], experiments[8]]
    else:
        indices = [int(x) - 1 for x in args.exp.split(",")]
        to_run = [experiments[i] for i in indices]

    total = len(to_run)
    print(f"\n🚀 Running {total} experiment(s)...")
    print(f"📁 Results saved to: {RESULTS_DIR}/")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for i, exp in enumerate(to_run):
        print(f"\n[{i+1}/{total}] ", end="")
        run_experiment(exp)

    print("\n\n" + "🏁 " * 20)
    print("  ALL EXPERIMENTS COMPLETE!")
    print("🏁 " * 20)

    print_results_table()
    generate_plots()
    list_prompts()

    print(f"\n📁 All outputs saved in: {RESULTS_DIR}/")
    print(f"   ├── results.json          (raw numbers)")
    print(f"   ├── logs/                  (full training logs)")
    print(f"   └── plots/                 (comparison charts)")
