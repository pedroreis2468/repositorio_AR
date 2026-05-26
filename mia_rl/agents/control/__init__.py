from .base import ControlAgent
from .linear_sarsa import LinearSarsaControl
from .monte_carlo import MonteCarloControl
from .reinforce import ReinforceAgent
from .sarsa import SarsaControl

try:
    from .torch_sarsa import TorchSarsaControl
except ImportError:
    pass  # PyTorch is an optional dependency
