"""
Double CCHP Q-Learning: combines Double Q-learning (van Hasselt 2010)
with the Cao-Chen-Hull-Poulos (2021) dual-Q formulation.

We maintain FOUR tables:
  Q1_A, Q1_B : two independent estimators of E[C_t | s, a]
  Q2_A, Q2_B : two independent estimators of E[C_t^2 | s, a]

At each step, flip a coin:
  Heads: use A-tables to SELECT a* = argmin (Q1_A + c*sqrt(max(0, Q2_A - Q1_A^2)))
         use B-tables to EVALUATE target values Q1_B[s', a*], Q2_B[s', a*]
         update A-tables
  Tails: symmetric, swap A and B

This breaks the "optimistic" correlation between action selection and
value estimation that causes vanilla Q-learning to overestimate.

Final policy uses the AVERAGE of A and B tables (standard practice).
"""

import numpy as np
import pandas as pd
import time
from scipy.stats import norm

# Import shared utilities from exp_framework
from exp_framework import (bs_price_vec, bs_delta_vec, simulate_paths_batch,
                          make_bs_policy, evaluate_policy, summarize)


def double_cchp_train(
    K, T, S0, sigma, r, dt, kappa, init_pos,
    actions, c_risk,
    n_time, n_money, n_gap, m_edges, g_edges,
    n_batches, batch_size,
    alpha_start, alpha_end, eps_start, eps_end,
    gamma=1.0, exploring_starts_frac=0.3,
    seed=8, print_every=50,
):
    """Double-CCHP Q-learning. Returns Q1_avg, Q2_avg, visits, cost_history."""
    rng = np.random.default_rng(seed)
    n_actions = len(actions)
    n_steps = int(round(T / dt))
    actions_arr = np.asarray(actions)

    Q1_A = np.zeros((n_time, n_money, n_gap, n_actions))
    Q1_B = np.zeros((n_time, n_money, n_gap, n_actions))
    Q2_A = np.zeros((n_time, n_money, n_gap, n_actions))
    Q2_B = np.zeros((n_time, n_money, n_gap, n_actions))
    visit_count = np.zeros((n_time, n_money, n_gap, n_actions), dtype=np.int32)
    cost_history = np.zeros(n_batches)

    phase1_end = int(exploring_starts_frac * n_batches)

    def tau_to_tidx(tau_arr):
        t_frac = 1.0 - tau_arr / T
        return np.clip(np.floor(t_frac * n_time).astype(int), 0, n_time - 1)

    def risk_adjusted_q(q1, q2):
        return q1 + c_risk * np.sqrt(np.maximum(q2 - q1**2, 0.0))

    for batch in range(n_batches):
        # Same schedule as single-CCHP
        if batch < phase1_end:
            p_frac = batch / max(1, phase1_end - 1)
            alpha = alpha_start + (alpha_start/2 - alpha_start) * p_frac
            eps = eps_start + (0.3 - eps_start) * p_frac
            S_init = S0 * np.exp(rng.normal(0, 0.15, size=batch_size))
            pos_init = rng.uniform(0, 1, size=batch_size)
            tau_init = rng.uniform(0.05, T, size=batch_size)
        else:
            p_frac = (batch - phase1_end) / max(1, n_batches - phase1_end - 1)
            alpha = alpha_start/2 + (alpha_end - alpha_start/2) * p_frac
            eps = 0.3 + (eps_end - 0.3) * p_frac
            S_init = np.full(batch_size, S0)
            pos_init = np.full(batch_size, init_pos)
            tau_init = np.full(batch_size, T)

        S = S_init.copy()
        pos = pos_init.copy()
        tau = tau_init.copy()
        ep_cost = np.zeros(batch_size)

        t_idx = tau_to_tidx(tau)
        m_idx = np.clip(np.digitize(S/K, m_edges) - 1, 0, n_money - 1)
        bsd = bs_delta_vec(S, K, tau, r, sigma)
        g_idx = np.clip(np.digitize(pos - bsd, g_edges) - 1, 0, n_gap - 1)

        alive = np.ones(batch_size, dtype=bool)

        for ti in range(n_steps):
            if not alive.any():
                break

            # Action selection: use AVERAGE of A and B for the behavior policy
            # (this is the standard Double-Q approach for eps-greedy exploration)
            q1_s = 0.5 * (Q1_A[t_idx, m_idx, g_idx] + Q1_B[t_idx, m_idx, g_idx])
            q2_s = 0.5 * (Q2_A[t_idx, m_idx, g_idx] + Q2_B[t_idx, m_idx, g_idx])
            ra_q = risk_adjusted_q(q1_s, q2_s)
            greedy = np.argmin(ra_q, axis=1)
            rand_a = rng.integers(n_actions, size=batch_size)
            explore = rng.random(batch_size) < eps
            a_idx = np.where(explore, rand_a, greedy)
            action = actions_arr[a_idx]

            # Env step (identical to single-CCHP)
            pos_prev = pos.copy(); S_prev = S.copy(); tau_prev = tau.copy()
            pos_next = np.clip(pos_prev + action, 0.0, 1.0)
            z = rng.standard_normal(batch_size)
            S_next = S_prev * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            tau_next = np.maximum(0.0, tau_prev - dt)
            C_prev = bs_price_vec(S_prev, K, tau_prev, r, sigma)
            C_next = bs_price_vec(S_next, K, tau_next, r, sigma)
            dPnL = (S_next - S_prev) * pos_prev \
                   - np.abs(pos_next - pos_prev) * S_next * kappa \
                   - C_next + C_prev
            cost = -dPnL / K
            done_now = tau_next <= 1e-12
            cost = np.where(done_now, cost + pos_next * S_next * kappa / K, cost)
            ep_cost = ep_cost + cost * alive

            # Next-state indices
            mon_next = S_next / K
            t_next = tau_to_tidx(tau_next)
            m_next = np.clip(np.digitize(mon_next, m_edges) - 1, 0, n_money - 1)
            bsd_next = bs_delta_vec(S_next, K, tau_next, r, sigma)
            g_next = np.clip(np.digitize(pos_next - bsd_next, g_edges) - 1, 0, n_gap - 1)

            # DOUBLE Q CORE: flip a coin per path to decide A vs B update
            update_A = rng.random(batch_size) < 0.5

            # --- Compute targets for ALL paths, using both (A-select + B-evaluate) and (B-select + A-evaluate) ---
            # A selects, B evaluates (used to update Q_A)
            q1_ns_A = Q1_A[t_next, m_next, g_next]
            q2_ns_A = Q2_A[t_next, m_next, g_next]
            ra_A = risk_adjusted_q(q1_ns_A, q2_ns_A)
            a_star_A = np.argmin(ra_A, axis=1)
            q1_eval_B = Q1_B[t_next, m_next, g_next, a_star_A]  # evaluate with B
            q2_eval_B = Q2_B[t_next, m_next, g_next, a_star_A]

            # B selects, A evaluates (used to update Q_B)
            q1_ns_B = Q1_B[t_next, m_next, g_next]
            q2_ns_B = Q2_B[t_next, m_next, g_next]
            ra_B = risk_adjusted_q(q1_ns_B, q2_ns_B)
            a_star_B = np.argmin(ra_B, axis=1)
            q1_eval_A = Q1_A[t_next, m_next, g_next, a_star_B]
            q2_eval_A = Q2_A[t_next, m_next, g_next, a_star_B]

            # CCHP targets (cross-evaluated):
            #   When updating A: use B's estimate of Q at (s', a*_A)
            q1_tgt_A = np.where(done_now, cost, cost + gamma * q1_eval_B)
            q2_tgt_A = np.where(done_now, cost**2,
                               cost**2 + 2*gamma*cost*q1_eval_B + gamma**2 * q2_eval_B)
            #   When updating B: use A's estimate of Q at (s', a*_B)
            q1_tgt_B = np.where(done_now, cost, cost + gamma * q1_eval_A)
            q2_tgt_B = np.where(done_now, cost**2,
                               cost**2 + 2*gamma*cost*q1_eval_A + gamma**2 * q2_eval_A)

            # Partition paths into those updating A vs B, apply only to alive paths
            alive_A = alive & update_A
            alive_B = alive & (~update_A)

            if alive_A.any():
                ti_a = t_idx[alive_A]; mi_a = m_idx[alive_A]
                gi_a = g_idx[alive_A]; ai_a = a_idx[alive_A]
                q1_cur = Q1_A[ti_a, mi_a, gi_a, ai_a]
                q2_cur = Q2_A[ti_a, mi_a, gi_a, ai_a]
                np.add.at(Q1_A, (ti_a, mi_a, gi_a, ai_a),
                         alpha * (q1_tgt_A[alive_A] - q1_cur))
                np.add.at(Q2_A, (ti_a, mi_a, gi_a, ai_a),
                         alpha * (q2_tgt_A[alive_A] - q2_cur))
                np.add.at(visit_count, (ti_a, mi_a, gi_a, ai_a), 1)

            if alive_B.any():
                ti_b = t_idx[alive_B]; mi_b = m_idx[alive_B]
                gi_b = g_idx[alive_B]; ai_b = a_idx[alive_B]
                q1_cur = Q1_B[ti_b, mi_b, gi_b, ai_b]
                q2_cur = Q2_B[ti_b, mi_b, gi_b, ai_b]
                np.add.at(Q1_B, (ti_b, mi_b, gi_b, ai_b),
                         alpha * (q1_tgt_B[alive_B] - q1_cur))
                np.add.at(Q2_B, (ti_b, mi_b, gi_b, ai_b),
                         alpha * (q2_tgt_B[alive_B] - q2_cur))
                np.add.at(visit_count, (ti_b, mi_b, gi_b, ai_b), 1)

            # Roll state
            S, pos, tau = S_next, pos_next, tau_next
            t_idx, m_idx, g_idx = t_next, m_next, g_next
            alive = alive & ~done_now

        cost_history[batch] = ep_cost.mean()

        if (batch + 1) % print_every == 0:
            recent = cost_history[max(0, batch - print_every + 1):batch + 1]
            print(f"Batch {batch+1:>5d}/{n_batches}  "
                  f"(eps={eps:.3f} alpha={alpha:.4f})  "
                  f"Avg cost/ep: {recent.mean():>8.4f}  "
                  f"Visit coverage: {100*np.count_nonzero(visit_count)/visit_count.size:.1f}%")

    # Return averaged Q-tables for use as the final policy
    Q1_avg = 0.5 * (Q1_A + Q1_B)
    Q2_avg = 0.5 * (Q2_A + Q2_B)
    return Q1_avg, Q2_avg, visit_count, cost_history, (Q1_A, Q1_B, Q2_A, Q2_B)
