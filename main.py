"""
Main entry point for the Q-learning hedging experiment.

Run this single file end-to-end to train one or more Q-learning agents and
produce the evaluation figures. Edit the CONFIG block below to control which
agents to train and what hyperparameters to use.

Usage:
    python main.py

This file orchestrates the other modules:
    env.py          - hedging environment
    qlearn.py       - tabular Q-learning agent
    q_train.py      - training loop and benchmark evaluators
    q_chunk.py      - chunked save/load training driver
    q_compare.py    - final evaluation and figure generation
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import HedgingEnv
from q_chunk import main as train_chunk, load
from q_compare import (
    evaluate_on_paths, make_q_action, bs_action, no_hedge_action,
    metrics,
)


# =============================================================================
# CONFIG -- edit these to control the experiment
# =============================================================================

# --- training budget ---------------------------------------------------------
# Total number of episodes per agent. 100k takes ~10 minutes on CPU per agent.
# Use 25000 for a quick test (~3 min/agent).
EPISODES_PER_AGENT = 100000

# Number of chunks the training is split into (purely a save-frequency knob).
# More chunks = more checkpoint files on disk; same total work.
NUM_CHUNKS = 2

# --- agents to train ---------------------------------------------------------
# Each entry: (checkpoint_name, kwargs for QAgent)
# Set this list to whichever agents you want to train and compare.
# To skip training of an agent that already has a saved checkpoint, comment it
# out below (the comparison will still load existing checkpoints by name).

AGENTS = [
    # (name,            kwargs passed to QAgent)
    ("q_baseline",      dict(risk_c=0.0)),
    ("q_double",        dict(risk_c=0.0, double_q=True)),
    ("q_c0.5",          dict(risk_c=0.5)),
    ("q_c1.0",          dict(risk_c=1.0)),
    ("q_c2.0",          dict(risk_c=2.0)),
]

# --- agent defaults (applied to every agent unless overridden in AGENTS) -----
AGENT_DEFAULTS = dict(
    state_dim=2,
    tau_bins=11,
    m_bins=11,
    h_bins=11,
    A=11,
    alpha=0.005,
    eps_decay_episodes=80000,
    alpha_decay_episodes=None,
    m_binning="uniform",
)

# --- environment defaults (the thesis setting) -------------------------------
# These are fixed by the thesis; rarely need to change.
ENV_PARAMS = dict(
    S0=100.0,
    K=100.0,
    T=1.0,
    N=252,
    sigma=0.20,
    r=0.0,
    kappa=0.01,
    reward_mode="apl",
)

# --- evaluation budget -------------------------------------------------------
N_EVAL_PATHS = 3000

# --- what to do --------------------------------------------------------------
DO_TRAIN     = True   # set to False to skip training (use existing checkpoints)
DO_EVALUATE  = True   # produce the table + main comparison figure
DO_PARETO    = True   # produce the mean-variance Pareto figure
OUTPUT_DIR   = "outputs"


# =============================================================================
# Execution
# =============================================================================
def train_all():
    """Train every agent in AGENTS, in chunks of EPISODES_PER_AGENT / NUM_CHUNKS."""
    chunk_size = EPISODES_PER_AGENT // NUM_CHUNKS
    for name, overrides in AGENTS:
        cfg = dict(AGENT_DEFAULTS)
        cfg.update(overrides)
        print("\n" + "=" * 70)
        print(f"Training '{name}' for {EPISODES_PER_AGENT} episodes "
              f"({NUM_CHUNKS} chunks of {chunk_size})")
        print(f"  config: {cfg}")
        print("=" * 70)
        # First chunk starts fresh; subsequent chunks resume.
        train_chunk(name, chunk_size, fresh=True, **cfg)
        for _ in range(NUM_CHUNKS - 1):
            train_chunk(name, chunk_size, fresh=False, **cfg)


def evaluate_all():
    """Evaluate every named agent on identical seeded paths, plus BS + no-hedge."""
    env = HedgingEnv(**ENV_PARAMS)
    p = env.option_premium
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Evaluating on {N_EVAL_PATHS} identical seeded paths")
    print("=" * 70)
    print(f"Option premium: {p:.4f}")

    results = {}
    # Benchmarks
    print("  evaluating BS delta...")
    c, d, s, nt = evaluate_on_paths(env, bs_action, N_EVAL_PATHS)
    results["BS delta"] = (c, d, s, nt)
    print("  evaluating no-hedge...")
    c, d, s, nt = evaluate_on_paths(env, no_hedge_action, N_EVAL_PATHS)
    results["No hedge"] = (c, d, s, nt)

    # Q-agents
    for name, _ in AGENTS:
        agent = load(name)
        if agent is None:
            print(f"  skip {name}: no checkpoint found")
            continue
        print(f"  evaluating {name}...")
        c, d, s, nt = evaluate_on_paths(env, make_q_action(agent), N_EVAL_PATHS)
        results[name] = (c, d, s, nt)

    # Print table
    print()
    print(f"{'Strategy':<22} {'Mean':>7} {'Mean%':>7} {'Std':>7} {'Std%':>7} "
          f"{'P95':>6} {'CVaR95':>7} {'sum|dH|':>8} {'#trades':>8}")
    print("-" * 100)
    for label, (c, d, s, nt) in results.items():
        m = metrics(c, p)
        print(f"{label:<22} {m['mean']:>7.3f} {m['mean_pct']:>6.2f}% "
              f"{m['std']:>7.3f} {m['std_pct']:>6.2f}% "
              f"{m['p95']:>6.2f} {m['cvar95']:>7.2f} "
              f"{float(np.mean(d)):>8.2f} {float(np.mean(nt)):>8.2f}")

    # Make comparison figure
    make_comparison_figure(results, p)
    return results


def make_comparison_figure(results, premium):
    """Hedge cost histogram + mean-std scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    ax = axes[0]
    all_costs = np.concatenate([c for c, _, _, _ in results.values()])
    bins = np.linspace(all_costs.min() - 1,
                       np.percentile(all_costs, 99) + 1, 60)
    for label, (c, _, _, _) in results.items():
        ax.hist(c, bins=bins, alpha=0.5, label=label, density=True)
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Hedge cost")
    ax.set_ylabel("Density")
    ax.set_title(f"Hedge cost distribution ({N_EVAL_PATHS} paths)")
    ax.legend(fontsize=8)

    # Mean-Std scatter
    ax = axes[1]
    for label, (c, _, _, _) in results.items():
        ax.scatter(np.std(c), np.mean(c), s=110, edgecolor="black",
                   linewidth=0.7, label=label)
        ax.annotate(label, (np.std(c), np.mean(c)),
                    xytext=(7, 7), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Std of hedge cost")
    ax.set_ylabel("Mean hedge cost")
    ax.set_title("Mean-Std tradeoff (lower-left = better)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "comparison.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved comparison figure to {out}")


def make_pareto_figure(results):
    """Plot mean vs std with all Q-agents and BS / no-hedge highlighted."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Q-agents (by name, in the order of AGENTS)
    q_xs, q_ys, q_labels = [], [], []
    for name, _ in AGENTS:
        if name in results:
            c = results[name][0]
            q_xs.append(float(np.std(c)))
            q_ys.append(float(np.mean(c)))
            q_labels.append(name)

    ax.plot(q_xs, q_ys, "o-", color="C3", markersize=10, lw=1.5,
            markeredgecolor="black", label="Q-learning variants")
    for x, y, lab in zip(q_xs, q_ys, q_labels):
        ax.annotate(lab, (x, y), xytext=(8, 6),
                    textcoords="offset points", fontsize=9)

    # BS
    if "BS delta" in results:
        c = results["BS delta"][0]
        ax.scatter([np.std(c)], [np.mean(c)], s=180, marker="*",
                   c="C0", edgecolor="black", zorder=4, label="BS delta")
        ax.annotate("BS", (np.std(c), np.mean(c)),
                    xytext=(-25, 6), textcoords="offset points", fontsize=10)

    # No hedge
    if "No hedge" in results:
        c = results["No hedge"][0]
        ax.scatter([np.std(c)], [np.mean(c)], s=120, marker="s",
                   c="gray", edgecolor="black", zorder=4, label="No hedge")
        ax.annotate("No hedge", (np.std(c), np.mean(c)),
                    xytext=(-55, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Std of hedge cost")
    ax.set_ylabel("Mean hedge cost")
    ax.set_title("Mean-variance Pareto frontier (lower-left = better)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "pareto.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved Pareto figure to {out}")


def main():
    if DO_TRAIN:
        train_all()
    results = None
    if DO_EVALUATE:
        results = evaluate_all()
    if DO_PARETO and results is not None:
        make_pareto_figure(results)
    print("\nAll done.")


if __name__ == "__main__":
    main()
