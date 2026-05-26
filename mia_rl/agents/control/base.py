from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable, TypeVar

from mia_rl.core.base import Agent, Transition

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)


class ControlAgent(Agent[StateT, ActionT], ABC):
    @abstractmethod
    def select_action(self, state: StateT) -> ActionT:
        raise NotImplementedError

    @abstractmethod
    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def action_value_of(self, state: StateT, action: ActionT) -> float:
        raise NotImplementedError

    def end_episode(self) -> None:
        """Optional hook called by training helpers when an episode boundary is reached."""
        return