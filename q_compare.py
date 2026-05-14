"""Final Q-learning comparison and figure generation."""
import os, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import HedgingEnv, bs_delta, bs_delta_scalar
from qlearn import QAgent
from q_train import evaluate_q
from q_chunk import load, make_env

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def evaluate_on_paths(env, action_fn, n_episodes, seed=9999):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_episodes)
    costs   = np.empty(n_episodes)
    sum_dH  = np.empty(n_episodes)   # total trade volume per episode
    n_trades = np.empty(n_episodes)  # count of rebalancing trades (|dH| > eps)
    final_S = np.empty(n_episodes)
    EPS = 1e-9
    for i, sd in enumerate(seeds):
        env.reset(seed=int(sd))
        H_prev = 0.0
        sdh = 0.0
        nt = 0
        done = False
        while not done:
            a = action_fn(env, H_prev)
            dH = abs(a - H_prev)
            sdh += dH
            if dH > EPS:
                nt += 1
            H_prev = a
            _, _, done, _ = env.step(a)
        costs[i] = env.hedge_cost
        sum_dH[i] = sdh
        n_trades[i] = nt
        final_S[i] = env.S
    return costs, sum_dH, final_S, n_trades


def bs_action(env, H_prev):
    tau = env.T - env.t * env.dt
    return bs_delta_scalar(env.S, env.K, env.r, tau, env.sigma)


def no_hedge_action(env, H_prev):
    return 0.0


def make_q_action(agent):
    def fn(env, H_prev):
        tau = env.T - env.t * env.dt
        m = env.S / env.K
        h = H_prev if agent.state_dim == 3 else 0.0
        _, a = agent.select_action(tau, m, h, greedy=True)
        return a
    return fn


def metrics(costs, premium):
    return {
        "mean": float(np.mean(costs)),
        "std":  float(np.std(costs)),
        "p95":  float(np.percentile(costs, 95)),
        "p99":  float(np.percentile(costs, 99)),
        "cvar95": float(np.mean(costs[costs >= np.percentile(costs, 95)])),
        "mean_pct": float(100 * np.mean(costs) / premium),
        "std_pct":  float(100 * np.std(costs) / premium),
    }


def main(n_eval=3000):
    env = make_env()
    p = env.option_premium
    print(f"Option premium: {p:.4f}")
    print(f"Evaluating on {n_eval} identical seeded paths.\n")

    results = {}
    bs_costs, bs_dH, bs_ST, bs_nt = evaluate_on_paths(env, bs_action, n_eval)
    results["BS delta"] = (bs_costs, bs_dH, bs_ST, bs_nt)
    nh_costs, nh_dH, nh_ST, nh_nt = evaluate_on_paths(env, no_hedge_action, n_eval)
    results["No hedge"] = (nh_costs, nh_dH, nh_ST, nh_nt)

    # Load Q-learning agents (trained with APL reward, thesis eq. 8)
    for name, label in [
        ("q2d_apl_v2",     "Q-learning"),
        ("q2d_improved",   "Q-learning (R-M α, log m bins)"),
        ("q2d_double_apl", "Double Q-learning"),
    ]:
        agent = load(name)
        if agent is None:
            print(f"  skip {name} (no checkpoint)")
            continue
        c, d, s, nt = evaluate_on_paths(env, make_q_action(agent), n_eval)
        results[label] = (c, d, s, nt)

    # Print table
    print(f"\n{'Strategy':<28} {'Mean':>7} {'Mean%':>7} {'Std':>7} {'Std%':>7}"
          f" {'P95':>6} {'CVaR95':>7} {'sum|dH|':>8} {'#trades':>8}")
    print("-" * 105)
    for label, (c, d, s, nt) in results.items():
        m = metrics(c, p)
        sdH = float(np.mean(d))
        n_trd = float(np.mean(nt))
        print(f"{label:<28} {m['mean']:>7.3f} {m['mean_pct']:>6.2f}% "
              f"{m['std']:>7.3f} {m['std_pct']:>6.2f}% "
              f"{m['p95']:>6.2f} {m['cvar95']:>7.2f} {sdH:>8.2f} {n_trd:>8.2f}")

    # Save metrics
    with open(os.path.join(OUT_DIR, "q_metrics.json"), "w") as f:
        out = {l: metrics(c, p) | {"sum_dH_mean": float(np.mean(d))}
               for l, (c, d, _, _) in results.items()}
        json.dump(out, f, indent=2)

    # Figure
    palette = {
        "BS delta":                       "#1f77b4",
        "No hedge":                       "#7f7f7f",
        "Q-learning":                     "#d62728",
        "Q-learning (R-M α, log m bins)": "#ff7f0e",
        "Double Q-learning":              "#2ca02c",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1) Histogram
    ax = axes[0, 0]
    bins = np.linspace(min(c.min() for c, _, _, _ in results.values()) - 1,
                       max(np.percentile(c, 99) for c, _, _, _ in results.values()) + 1,
                       60)
    for label, (c, _, _, _) in results.items():
        ax.hist(c, bins=bins, alpha=0.5, label=label,
                color=palette.get(label, None), density=True)
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Hedge cost (currency)")
    ax.set_ylabel("Density")
    ax.set_title(f"Hedge cost distribution ({n_eval} paths)")
    ax.legend(fontsize=9)

    # 2) Mean-Std scatter
    ax = axes[0, 1]
    for label, (c, _, _, _) in results.items():
        ax.scatter(np.std(c), np.mean(c), s=100, color=palette.get(label, None),
                   label=label, edgecolor="black", linewidth=0.7)
        ax.annotate(label, (np.std(c), np.mean(c)),
                    xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Std of hedge cost")
    ax.set_ylabel("Mean hedge cost")
    ax.set_title("Mean–Std tradeoff (lower-left = better)")
    ax.grid(True, alpha=0.3)

    # 3) Trading intensity: show both #trades and sum|dH| as grouped bars
    ax = axes[1, 0]
    labels = list(results.keys())
    sumdH_means  = [np.mean(results[l][1]) for l in labels]
    ntrade_means = [np.mean(results[l][3]) for l in labels]
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w/2, ntrade_means, w, color=[palette.get(l, "gray") for l in labels],
           edgecolor="black", label="# trades (steps with ΔH≠0)")
    ax.bar(x + w/2, sumdH_means, w, color=[palette.get(l, "gray") for l in labels],
           edgecolor="black", alpha=0.45, label="sum |ΔH|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("per episode")
    ax.set_title("Trading intensity")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    # 4) Cost vs final S
    ax = axes[1, 1]
    for label in ["BS delta", "Q-learning", "Q-learning (R-M α, log m bins)",
                  "Double Q-learning"]:
        if label not in results:
            continue
        c, _, sT, _ = results[label]
        bin_edges = np.linspace(70, 140, 25)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        binned = np.zeros(len(bin_centers))
        cnt = np.zeros(len(bin_centers))
        for i, st in enumerate(sT):
            j = np.searchsorted(bin_edges, st) - 1
            if 0 <= j < len(bin_centers):
                binned[j] += c[i]; cnt[j] += 1
        with np.errstate(invalid="ignore"):
            ax.plot(bin_centers, binned / np.maximum(cnt, 1),
                    label=label, color=palette[label], lw=2)
    ax.axvline(env.K, color="black", ls=":", lw=0.8, label="Strike")
    ax.set_xlabel("Final stock price S_T")
    ax.set_ylabel("Mean hedge cost | S_T")
    ax.set_title("Conditional cost by terminal price")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("Q-learning vs. Double Q-learning vs. Black–Scholes (T=1y, N=252, κ=1%)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "q_comparison.png"), dpi=140, bbox_inches="tight")
    print(f"\nSaved figure to {OUT_DIR}/q_comparison.png")

    # Plot policy heatmap (BS vs Q vs improved Q vs Double Q)
    fig2, axes = plt.subplots(1, 4, figsize=(22, 5))
    ms = np.linspace(0.6, 1.4, 80)
    taus = np.linspace(1.0, 0.005, 60)
    bs_grid = np.zeros((len(taus), len(ms)))
    for i, tau in enumerate(taus):
        for j, m in enumerate(ms):
            bs_grid[i, j] = float(bs_delta(m * 100, 100, 0.0, tau, 0.20))

    im = axes[0].imshow(bs_grid, origin="upper", aspect="auto",
                        extent=[ms.min(), ms.max(), taus.min(), taus.max()],
                        vmin=0, vmax=1, cmap="viridis")
    axes[0].set_xlabel("Moneyness S/K"); axes[0].set_ylabel("Time to maturity τ")
    axes[0].set_title("BS delta")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    for ax_i, ckpt_name, title in [
        (1, "q2d_apl_v2",     "Q-learning"),
        (2, "q2d_improved",   "Q-learning (R-M α, log m)"),
        (3, "q2d_double_apl", "Double Q-learning"),
    ]:
        agent = load(ckpt_name)
        if agent is None:
            continue
        q_grid = np.zeros((len(taus), len(ms)))
        for i, tau in enumerate(taus):
            for j, m in enumerate(ms):
                q_grid[i, j] = agent.greedy_action(float(tau), float(m), 0.0)
        im2 = axes[ax_i].imshow(q_grid, origin="upper", aspect="auto",
                             extent=[ms.min(), ms.max(), taus.min(), taus.max()],
                             vmin=0, vmax=1, cmap="viridis")
        axes[ax_i].set_xlabel("Moneyness S/K"); axes[ax_i].set_ylabel("Time to maturity τ")
        axes[ax_i].set_title(title)
        plt.colorbar(im2, ax=axes[ax_i], fraction=0.046)

    fig2.suptitle("Hedge ratio H(τ, m): BS vs three Q-learning variants",
                  fontsize=12)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "q_policy.png"), dpi=140, bbox_inches="tight")
    print(f"Saved policy heatmap to {OUT_DIR}/q_policy.png")


if __name__ == "__main__":
    main(n_eval=3000)
