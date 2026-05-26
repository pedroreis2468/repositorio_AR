from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Hashable, Iterable, TypeVar

State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action")


class TabularMDP(ABC, Generic[State, Action]):
    """Abstract interface for known-model finite MDPs used in dynamic programming."""

    @abstractmethod
    def states(self) -> list[State]:
        raise NotImplementedError

    @abstractmethod
    def possible_actions(self, state: State) -> list[Action]:
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        raise NotImplementedError

    @abstractmethod
    def transitions(
        self,
        state: State,
        action: Action,
    ) -> Iterable[tuple[float, State, float, bool]]:
        """Return `(probability, next_state, reward, done)` tuples."""
        raise NotImplementedError
