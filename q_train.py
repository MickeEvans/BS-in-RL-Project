"""Training and evaluation for the unified Q-learning agent.
Both 2D-state and 3D-state versions live in qlearn.QAgent."""
import time, json
import numpy as np
from env import HedgingEnv, bs_delta_scalar
from qlearn import QAgent


# -----------------------------------------------------------------------------
# Benchmark policies (BS delta hedge, no-hedge)
# -----------------------------------------------------------------------------
def evaluate_bs(env, n_episodes=2000, seed=999):
    """Roll out the BS delta hedge policy."""
    rng = np.random.default_rng(seed)
    costs = []
    for _ in range(n_episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        while not done:
            tau = env.T - env.t * env.dt
            d = bs_delta_scalar(env.S, env.K, env.r, tau, env.sigma)
            _, _, done, _ = env.step(d)
        costs.append(env.hedge_cost)
    return np.array(costs)


def evaluate_no_hedge(env, n_episodes=2000, seed=999):
    """Roll out the do-nothing (H=0 forever) policy."""
    rng = np.random.default_rng(seed)
    costs = []
    for _ in range(n_episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        while not done:
            _, _, done, _ = env.step(0.0)
        costs.append(env.hedge_cost)
    return np.array(costs)


def train_q(env, agent, n_episodes, log_every=2000, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    log = []
    running = []
    for ep in range(n_episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        tau = env.T
        m = env.S / env.K
        h = 0.0  # current hedge holding (env starts with H=0)
        while not done:
            a_idx, a = agent.select_action(tau, m, h, greedy=False)
            obs_next, reward, done, _ = env.step(a)
            tau_n, m_n = float(obs_next[0]), float(obs_next[1])
            h_n = a   # after step, the hedge ratio in env is the action just taken
                      # (env auto-liquidates at done, but for the (s,a,r,s') tuple
                      #  we pass the pre-liquidation H_next which equals the action)
            if done:
                # at terminal, the env zeroes H during step; for the bootstrap target
                # we don't need a meaningful s' so just use the same idx; done=True
                # routes target to just `r`.
                pass
            agent.update(tau, m, h, a_idx, reward, tau_n, m_n, h_n, done)
            tau, m, h = tau_n, m_n, h_n
        agent.end_episode()
        running.append(env.hedge_cost)
        if (ep + 1) % log_every == 0:
            mc = float(np.mean(running[-log_every:]))
            td_recent = float(np.mean(agent.td_history[-log_every:])) \
                          if agent.td_history else 0.0
            log.append((ep + 1, mc, agent.epsilon, agent.alpha, td_recent))
            if verbose:
                print(f"  ep={ep+1:6d}  eps={agent.epsilon:.3f}  "
                      f"alpha={agent.alpha:.4f}  "
                      f"mean_cost(last {log_every})={mc:7.3f}  "
                      f"|TD|={td_recent:.3f}")
    return log


def evaluate_q(env, agent, n_episodes=2000, seed=999):
    rng = np.random.default_rng(seed)
    costs = np.empty(n_episodes)
    sum_dH = np.empty(n_episodes)
    for ep in range(n_episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        tau, m, h = env.T, env.S / env.K, 0.0
        H_prev, sdh = 0.0, 0.0
        while not done:
            _, a = agent.select_action(tau, m, h, greedy=True)
            sdh += abs(a - H_prev); H_prev = a
            obs_next, _, done, _ = env.step(a)
            tau, m = float(obs_next[0]), float(obs_next[1])
            h = a
        costs[ep] = env.hedge_cost
        sum_dH[ep] = sdh
    return costs, sum_dH


def run(name, env_kwargs=None, agent_kwargs=None, n_episodes=20000,
        verbose=True):
    env_kwargs = env_kwargs or {}
    agent_kwargs = agent_kwargs or {}

    env_defaults = dict(S0=100, K=100, T=1.0, N=252, sigma=0.20,
                        r=0.0, kappa=0.01, seed=0, reward_mode='apl')
    env_defaults.update(env_kwargs)
    env = HedgingEnv(**env_defaults)
    p = env.option_premium

    agent_defaults = dict(state_dim=3, tau_bins=26, m_bins=21, h_bins=11, A=11,
                          alpha=0.1, eps_decay_episodes=15000)
    agent_defaults.update(agent_kwargs)
    agent = QAgent(N=env.N, dt=env.dt, **agent_defaults)

    if verbose:
        print(f"\n=== {name} ===")
        print(f"  env: kappa={env.kappa}, N={env.N}, reward={env.reward_mode}")
        print(f"  agent: state_dim={agent.state_dim} "
              f"shape={agent.Q.shape} alpha={agent.alpha} "
              f"double_q={agent.double_q} risk_c={agent.risk_c}")

    t0 = time.time()
    log = train_q(env, agent, n_episodes, log_every=n_episodes // 5,
                  verbose=verbose)
    elapsed = time.time() - t0

    costs, sum_dH = evaluate_q(env, agent, n_episodes=2000, seed=9999)
    res = {
        "name": name,
        "config": agent_defaults,
        "n_episodes": n_episodes,
        "elapsed_s": round(elapsed, 1),
        "mean": float(costs.mean()),
        "std":  float(costs.std()),
        "p95":  float(np.percentile(costs, 95)),
        "cvar95": float(np.mean(costs[costs >= np.percentile(costs, 95)])),
        "mean_pct": float(100 * costs.mean() / p),
        "std_pct":  float(100 * costs.std() / p),
        "sum_dH":   float(sum_dH.mean()),
        "premium":  float(p),
    }
    if verbose:
        print(f"  trained in {elapsed:.1f}s")
        print(f"  EVAL  mean={res['mean']:7.3f} ({res['mean_pct']:6.2f}%)  "
              f"std={res['std']:6.3f} ({res['std_pct']:6.2f}%)")
        print(f"        P95={res['p95']:6.2f}  CVaR95={res['cvar95']:6.2f}  "
              f"sum|dH|={res['sum_dH']:5.2f}")
    return res, agent


if __name__ == "__main__":
    pass
