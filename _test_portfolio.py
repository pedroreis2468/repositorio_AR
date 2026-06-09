"""Smoke test for the consolidated mia_rl portfolio package."""
import sys
import pathlib

# Add portefolio/ to path so we can import mia_rl directly
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "portefolio"))

errors = []

def test(name, fn):
    try:
        fn()
        print(f"  [OK]  {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        errors.append(name)

# ── 1. Core imports ───────────────────────────────────────────────────────────
def t_import_core():
    import mia_rl
    from mia_rl.core.base import Agent, Episode, Environment, Policy, Transition

def t_import_agents():
    from mia_rl.agents.prediction.monte_carlo import FirstVisitMonteCarloPrediction
    from mia_rl.agents.prediction.td import TD0Prediction
    from mia_rl.agents.prediction.td_n import NStepTDPrediction
    from mia_rl.agents.control.sarsa import SarsaControl
    from mia_rl.agents.control.n_step_sarsa import NStepSarsaControl
    from mia_rl.agents.control.reinforce import ReinforceAgent
    from mia_rl.agents.planning.mcts import MCTSAgent, MCTSNode
    # torch_sarsa is optional (requires PyTorch in the environment)
    try:
        from mia_rl.agents.control.torch_sarsa import TorchSarsaControl
    except ImportError:
        print("    (torch_sarsa skipped — PyTorch not installed)")

def t_import_envs():
    from mia_rl.envs.blackjack import BlackjackEnv
    from mia_rl.envs.windy_gridworld import WindyGridworldEnv
    from mia_rl.envs.tictactoe import TicTacToeEnv

def t_import_features():
    from mia_rl.features.windy_gridworld import tile_features
    from mia_rl.features.tictactoe import encode_state

def t_import_experiments():
    from mia_rl.experiments.training import generate_episode
    from mia_rl.experiments.reinforce_tictactoe import run_reinforce_episode
    from mia_rl.experiments.mcts_tictactoe import make_mcts_policy

def t_import_kbandits():
    from mia_rl.scripts.run_kbandits import (
        KArmedBandit, EpsilonGreedy, UCB, GradientBandit, run_experiment
    )

# ── 2. MCTS functional test ───────────────────────────────────────────────────
def t_mcts_functional():
    from mia_rl.agents.planning.mcts import MCTSAgent
    from mia_rl.envs.tictactoe import TicTacToeEnv

    env = TicTacToeEnv()
    agent = MCTSAgent(n_simulations=50)
    state = env.reset()
    action = agent.select_action(state, env.current_player)
    assert 0 <= action <= 8, f"invalid action {action}"

# ── 3. K-bandits functional test ─────────────────────────────────────────────
def t_kbandits_functional():
    from mia_rl.scripts.run_kbandits import (
        KArmedBandit, EpsilonGreedy, UCB, GradientBandit, run_experiment
    )
    env = KArmedBandit()
    for AgentCls, kwargs in [
        (EpsilonGreedy, {"epsilon": 0.1}),
        (UCB, {"c": 2.0}),
        (GradientBandit, {"alpha": 0.1}),
    ]:
        agent = AgentCls(**kwargs)
        r, o = run_experiment(agent, env, steps=20, runs=10)
        assert r.shape == (20,)

# ── 4. MC Prediction functional test ─────────────────────────────────────────
def t_mc_prediction():
    from mia_rl.envs.blackjack import BlackjackEnv
    from mia_rl.agents.prediction.monte_carlo import FirstVisitMonteCarloPrediction
    from mia_rl.experiments.training import generate_episode
    from mia_rl.policies.blackjack import ThresholdPolicy

    env = BlackjackEnv()
    agent = FirstVisitMonteCarloPrediction()
    agent.reset()
    policy = ThresholdPolicy(threshold=20)
    for _ in range(10):
        ep = generate_episode(env, policy)
        agent.update_episode(ep)

# ── 5. Experimental functional tests ─────────────────────────────────────────
def t_import_experimental():
    from mia_rl.experimental.double_q import DoubleQLearningControl
    from mia_rl.experimental.dqn import DQNAgent

def t_double_q_functional():
    from mia_rl.experimental.double_q import DoubleQLearningControl
    from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
    from mia_rl.core.base import Transition
    env = WindyGridworldEnv()
    agent = DoubleQLearningControl(actions=ACTIONS, seed=42)
    agent.reset()
    state = env.reset()
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    agent.update_transition(Transition(state, action, reward, next_state, done))

def t_dqn_functional():
    import numpy as np
    from mia_rl.experimental.dqn import DQNAgent
    agent = DQNAgent(n_features=27, n_actions=9, batch_size=2, target_update_every=5, seed=42)
    agent.reset()
    phi = np.zeros(27, dtype=np.float32)
    available = [0, 1, 2]
    action = agent.select_action(phi, available)
    assert action in available
    agent.store_transition(phi, action, 1.0, phi, False)
    agent.store_transition(phi, action, -1.0, None, True)
    loss = agent.update_model()
    assert loss is not None

def t_n_step_sarsa_functional():
    from mia_rl.agents.control.n_step_sarsa import NStepSarsaControl
    from mia_rl.envs.windy_gridworld import ACTIONS, WindyGridworldEnv
    from mia_rl.core.base import Transition
    env = WindyGridworldEnv()
    agent = NStepSarsaControl(actions=ACTIONS, n_steps=3, seed=42)
    agent.reset()
    state = env.reset()
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    agent.update_transition(Transition(state, action, reward, next_state, done))

def t_n_step_td_functional():
    from mia_rl.envs.blackjack import BlackjackEnv
    from mia_rl.agents.prediction.td_n import NStepTDPrediction
    from mia_rl.experiments.training import generate_episode
    from mia_rl.policies.blackjack import ThresholdPolicy

    env = BlackjackEnv()
    agent = NStepTDPrediction(n=3)
    agent.reset()
    policy = ThresholdPolicy(threshold=20)
    for _ in range(5):
        ep = generate_episode(env, policy)
        agent.update_episode(ep)

# ── Run all ───────────────────────────────────────────────────────────────────
print("\nmia_rl Portfolio — Smoke Tests")
print("=" * 40)
test("Core imports",           t_import_core)
test("Agent imports",          t_import_agents)
test("Environment imports",    t_import_envs)
test("Feature imports",        t_import_features)
test("Experiment imports",     t_import_experiments)
test("K-Bandits script import",t_import_kbandits)
test("MCTS functional",        t_mcts_functional)
test("K-Bandits functional",   t_kbandits_functional)
test("MC prediction functional", t_mc_prediction)
test("Experimental imports",   t_import_experimental)
test("Double Q functional",    t_double_q_functional)
test("DQN functional",         t_dqn_functional)
test("n-step SARSA functional", t_n_step_sarsa_functional)
test("n-step TD functional",    t_n_step_td_functional)

print("=" * 40)
if errors:
    print(f"FAILED: {len(errors)} test(s) — {errors}")
    sys.exit(1)
else:
    print(f"All tests passed!")
