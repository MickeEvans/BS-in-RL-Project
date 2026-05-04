"""
Q-Learning agents — warm-start initialisation and training loops.

Two variants:
  1. Standard tabular Q-learning (train)
  2. Double Q-learning (train_double) — Hasselt 2010 ablation

Reward function (the heart of the RL formulation)
--------------------------------------------------
At each step the agent receives:

    reward = -κ · |trade| · S                           (transaction cost penalty)
             -λ · (pos − δ)² · S² · σ² · dt            (variance / mis-hedge penalty)

At the final step, the option payoff is also subtracted:

    reward -= max(S_T − K, 0)

The hyperparameter λ (risk_lambda) controls the risk-cost trade-off:
  - Low λ  → agent minimises costs, ignores hedging risk (under-hedges)
  - High λ → agent tracks delta tightly, high cost (approaches BS)
  - Sweet spot (~0.3) → agent discovers a no-trade band that beats BS
"""

import numpy as np
from config import N, K, dt, sigma, kappa
from environment import (
    N_T, N_M, N_E, N_A, ACTIONS, E_EDGES,
    get_state, sim_paths,
)


# ══════════════════════════════════════════════════════════════════════════════
# BS-HEURISTIC WARM START
# ══════════════════════════════════════════════════════════════════════════════

def warm_init(Q):
    """
    Seed the Q-table with a Black-Scholes heuristic.

    For each state-action cell, set Q to: "prefer actions that reduce
    position error, with a small penalty for large trades."

    This is NOT the correct Q-function — it knows nothing about
    transaction costs or time structure — but it gives the agent a
    sensible starting direction. Training converges in ~15k episodes
    instead of ~200k+ from cold start. About 10× speedup.
    """
    for t_i in range(N_T):
        for m_i in range(N_M):
            for e_i in range(N_E):
                err_mid = (E_EDGES[e_i] + E_EDGES[e_i + 1]) / 2.0
                for a_i, da in enumerate(ACTIONS):
                    Q[t_i, m_i, e_i, a_i] = -abs(err_mid - da) - 0.5 * abs(da)


# ══════════════════════════════════════════════════════════════════════════════
# STANDARD Q-LEARNING
# ══════════════════════════════════════════════════════════════════════════════

def train(Q, visit, n_episodes, risk_lambda,
          lr0=0.15, gamma=1.0,
          eps_start=0.5, eps_end=0.01, eps_decay=None,
          seed=0, n_parallel=64):
    """
    Train a tabular Q-learning agent.

    Update rule (Bellman):
        Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]

    The learning rate α decays with the visit count per cell, so
    well-explored states stabilise early.

    Parameters
    ----------
    Q            : np.ndarray shape (N_T, N_M, N_E, N_A) — Q-table (modified in-place)
    visit        : np.ndarray same shape, int — visit counters (modified in-place)
    n_episodes   : int   — total training episodes
    risk_lambda  : float — λ, the risk-aversion coefficient
    lr0          : float — initial learning rate
    gamma        : float — discount factor (1.0 = undiscounted)
    eps_start    : float — initial exploration rate
    eps_end      : float — final exploration rate
    eps_decay    : float — per-episode decay multiplier (auto-computed if None)
    seed         : int   — random seed for reproducibility
    n_parallel   : int   — number of paths simulated per batch
    """
    if eps_decay is None:
        eps_decay = (eps_end / eps_start) ** (1.0 / n_episodes)
    eps = eps_start
    rng = np.random.default_rng(seed)
    no_op_action = int(np.argmin(np.abs(ACTIONS)))

    n_batches = n_episodes // n_parallel

    for batch in range(n_batches):
        paths = sim_paths(n_parallel, seed=seed * 997 + batch)
        pos   = np.zeros(n_parallel)

        for step in range(N):
            S       = paths[:, step]
            S_next  = paths[:, step + 1]
            is_last = (step == N - 1)

            t_idx, m_idx, e_idx, delta = get_state(S, pos, step)

            # ── Action selection (ε-greedy) ──────────────────────────────
            if is_last:
                a_idx   = np.full(n_parallel, no_op_action, dtype=int)
                new_pos = np.zeros(n_parallel)
            else:
                explore  = rng.random(n_parallel) < eps
                q_vals   = Q[t_idx, m_idx, e_idx, :]          # shape (P, A)
                greedy_a = np.argmax(q_vals, axis=1)
                rand_a   = rng.integers(0, N_A, n_parallel)
                a_idx    = np.where(explore, rand_a, greedy_a)
                new_pos  = np.clip(pos + ACTIONS[a_idx], 0.0, 1.0)

            # ── Reward ───────────────────────────────────────────────────
            trade  = new_pos - pos
            reward = -kappa * np.abs(trade) * S                      # TC penalty

            if risk_lambda > 0 and not is_last:
                pos_err = new_pos - delta
                reward -= risk_lambda * (pos_err**2) * (S**2) * sigma**2 * dt  # variance penalty

            if is_last:
                reward -= np.maximum(S_next - K, 0.0)               # option payoff

            # ── Bootstrap target ─────────────────────────────────────────
            if not is_last:
                tn_idx, mn_idx, en_idx, _ = get_state(S_next, new_pos, step + 1)
                best_next = np.max(Q[tn_idx, mn_idx, en_idx, :], axis=1)
                targets   = reward + gamma * best_next
            else:
                targets = reward

            # ── Q-update (per-path, adaptive learning rate) ──────────────
            for p in range(n_parallel):
                ti, mi, ei, ai = t_idx, m_idx[p], e_idx[p], a_idx[p]
                visit[ti, mi, ei, ai] += 1
                lr = lr0 / (1 + 0.0005 * visit[ti, mi, ei, ai])
                Q[ti, mi, ei, ai] += lr * (targets[p] - Q[ti, mi, ei, ai])

            pos = new_pos

        # Decay exploration rate
        for _ in range(n_parallel):
            eps = max(eps_end, eps * eps_decay)


# ══════════════════════════════════════════════════════════════════════════════
# DOUBLE Q-LEARNING  (Hasselt, 2010)
# ══════════════════════════════════════════════════════════════════════════════
#
# Standard Q-learning has maximisation bias: E[max Q̂] ≥ max E[Q̂].
# Double Q fixes this with two tables — one picks the action, the other
# evaluates it, so the biases cancel.
#
# In this problem Double Q is actually WORSE because:
#   1. Each table gets ~half the updates → halved effective training budget
#   2. The reward is mostly deterministic (TC + payoff), so there's little
#      overestimation bias to fix
#   3. The warm start already gives sensible Q-values
#
# Included as an ablation / negative result for the thesis.
# ══════════════════════════════════════════════════════════════════════════════

def train_double(Q_A, Q_B, visit_A, visit_B, n_episodes, risk_lambda,
                 lr0=0.15, gamma=1.0,
                 eps_start=0.5, eps_end=0.01, eps_decay=None,
                 seed=0, n_parallel=64):
    """
    Double Q-learning. Action selection uses Q_A + Q_B (equivalent to
    averaging for argmax). Each update randomly picks one table to update.
    """
    if eps_decay is None:
        eps_decay = (eps_end / eps_start) ** (1.0 / n_episodes)
    eps = eps_start
    rng = np.random.default_rng(seed)
    no_op_action = int(np.argmin(np.abs(ACTIONS)))

    for batch in range(n_episodes // n_parallel):
        paths = sim_paths(n_parallel, seed=seed * 997 + batch)
        pos   = np.zeros(n_parallel)

        for step in range(N):
            S       = paths[:, step]
            S_next  = paths[:, step + 1]
            is_last = (step == N - 1)
            t_idx, m_idx, e_idx, delta = get_state(S, pos, step)

            if is_last:
                a_idx   = np.full(n_parallel, no_op_action, dtype=int)
                new_pos = np.zeros(n_parallel)
            else:
                explore  = rng.random(n_parallel) < eps
                q_sum    = Q_A[t_idx, m_idx, e_idx, :] + Q_B[t_idx, m_idx, e_idx, :]
                greedy_a = np.argmax(q_sum, axis=1)
                rand_a   = rng.integers(0, N_A, n_parallel)
                a_idx    = np.where(explore, rand_a, greedy_a)
                new_pos  = np.clip(pos + ACTIONS[a_idx], 0.0, 1.0)

            trade  = new_pos - pos
            reward = -kappa * np.abs(trade) * S
            if risk_lambda > 0 and not is_last:
                pos_err = new_pos - delta
                reward -= risk_lambda * (pos_err**2) * (S**2) * sigma**2 * dt
            if is_last:
                reward -= np.maximum(S_next - K, 0.0)

            if not is_last:
                tn_idx, mn_idx, en_idx, _ = get_state(S_next, new_pos, step + 1)

            for p in range(n_parallel):
                ti, mi, ei, ai = t_idx, m_idx[p], e_idx[p], a_idx[p]
                update_A = rng.random() < 0.5

                if update_A:
                    if not is_last:
                        a_star = int(np.argmax(Q_A[tn_idx, mn_idx[p], en_idx[p], :]))
                        tgt = reward[p] + gamma * Q_B[tn_idx, mn_idx[p], en_idx[p], a_star]
                    else:
                        tgt = reward[p]
                    visit_A[ti, mi, ei, ai] += 1
                    lr = lr0 / (1 + 0.0005 * visit_A[ti, mi, ei, ai])
                    Q_A[ti, mi, ei, ai] += lr * (tgt - Q_A[ti, mi, ei, ai])
                else:
                    if not is_last:
                        a_star = int(np.argmax(Q_B[tn_idx, mn_idx[p], en_idx[p], :]))
                        tgt = reward[p] + gamma * Q_A[tn_idx, mn_idx[p], en_idx[p], a_star]
                    else:
                        tgt = reward[p]
                    visit_B[ti, mi, ei, ai] += 1
                    lr = lr0 / (1 + 0.0005 * visit_B[ti, mi, ei, ai])
                    Q_B[ti, mi, ei, ai] += lr * (tgt - Q_B[ti, mi, ei, ai])

            pos = new_pos

        for _ in range(n_parallel):
            eps = max(eps_end, eps * eps_decay)
