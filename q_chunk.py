"""Chunked Q-learning training that saves the Q-table to disk so we can
resume across multiple bash invocations. Mirrors train_named.py but for
the tabular Q-agent."""
import os, time, pickle, argparse
import numpy as np
from env import HedgingEnv
from qlearn import QAgent
from q_train import train_q, evaluate_q

CKPT_DIR = "q_ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)


def make_env():
    return HedgingEnv(S0=100, K=100, T=1.0, N=252, sigma=0.20,
                      r=0.0, kappa=0.01, seed=0, reward_mode='apl')


def make_agent(**kwargs):
    defaults = dict(state_dim=2, tau_bins=11, m_bins=11, h_bins=11, A=11,
                    alpha=0.02, eps_decay_episodes=30000)
    defaults.update(kwargs)
    env = make_env()
    return QAgent(N=env.N, dt=env.dt, **defaults)


def ckpt_path(name):
    return (os.path.join(CKPT_DIR, name + ".npy"),
            os.path.join(CKPT_DIR, name + ".meta"))


def save(agent, name):
    p_q, p_meta = ckpt_path(name)
    if agent.double_q:
        np.save(p_q, np.stack([agent.Q, agent.Q2]))
    else:
        np.save(p_q, agent.Q)
    with open(p_meta, "wb") as f:
        pickle.dump({
            "episode": agent.episode,
            "td_history": agent.td_history,
            "config": dict(state_dim=agent.state_dim, tau_bins=agent.tau_bins,
                           m_bins=agent.m_bins, h_bins=agent.h_bins, A=agent.A,
                           alpha=agent.alpha_0,
                           alpha_decay_episodes=agent.alpha_decay_episodes,
                           m_binning=agent.m_binning,
                           eps_decay_episodes=agent.eps_decay_episodes,
                           eps_start=agent.eps_start, eps_end=agent.eps_end,
                           gamma=agent.gamma, risk_c=agent.risk_c,
                           double_q=agent.double_q),
        }, f)


def load(name):
    p_q, p_meta = ckpt_path(name)
    if not os.path.exists(p_q):
        return None
    with open(p_meta, "rb") as f:
        meta = pickle.load(f)
    agent = make_agent(**meta["config"])
    arr = np.load(p_q)
    if agent.double_q:
        agent.Q  = arr[0].copy()
        agent.Q2 = arr[1].copy()
    else:
        agent.Q = arr.copy()
    agent.episode = meta["episode"]
    agent.td_history = list(meta.get("td_history", []))
    return agent


def main(name, chunk_eps, fresh=False, **kwargs):
    if fresh:
        for p in ckpt_path(name):
            if os.path.exists(p):
                os.remove(p)
    agent = load(name)
    if agent is None:
        agent = make_agent(**kwargs)
        print(f"[{name}] Fresh start. Q.shape={agent.Q.shape} alpha={agent.alpha}")
    else:
        print(f"[{name}] Resumed at ep={agent.episode}, eps={agent.epsilon:.3f}")

    env = make_env()
    t0 = time.time()
    train_q(env, agent, chunk_eps, log_every=max(1, chunk_eps // 5), verbose=True)
    print(f"[{name}] Chunk: {time.time()-t0:.1f}s, total ep={agent.episode}")

    # Quick eval
    costs, dH = evaluate_q(env, agent, n_episodes=1000, seed=9999)
    p = env.option_premium
    print(f"  EVAL  mean={costs.mean():.3f} ({100*costs.mean()/p:.2f}%)  "
          f"std={costs.std():.3f} ({100*costs.std()/p:.2f}%)  sum|dH|={dH.mean():.2f}")

    save(agent, name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name", type=str)
    p.add_argument("chunk_eps", type=int)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--alpha", type=float, default=0.005)
    p.add_argument("--alpha_decay_episodes", type=int, default=None,
                   help="If set, use Robbins-Monro decay alpha_t = alpha/(1+ep/this).")
    p.add_argument("--m_binning", type=str, default="uniform",
                   choices=["uniform", "log_centered"])
    p.add_argument("--state_dim", type=int, default=2)
    p.add_argument("--tau_bins", type=int, default=11)
    p.add_argument("--m_bins", type=int, default=11)
    p.add_argument("--h_bins", type=int, default=11)
    p.add_argument("--A", type=int, default=11)
    p.add_argument("--eps_decay_episodes", type=int, default=80000)
    p.add_argument("--double_q", action="store_true")
    p.add_argument("--risk_c", type=float, default=0.0)
    a = p.parse_args()
    kwargs = dict(alpha=a.alpha, alpha_decay_episodes=a.alpha_decay_episodes,
                  m_binning=a.m_binning,
                  state_dim=a.state_dim, tau_bins=a.tau_bins,
                  m_bins=a.m_bins, h_bins=a.h_bins, A=a.A,
                  eps_decay_episodes=a.eps_decay_episodes, double_q=a.double_q,
                  risk_c=a.risk_c)
    main(a.name, a.chunk_eps, fresh=a.fresh, **kwargs)
