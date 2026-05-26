from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Hashable, Optional, TypeVar

State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action")


@dataclass(frozen=True)
class Transition(Generic[State, Action]):
    state: State
    action: Action
    reward: float
    next_state: Optional[State]
    done: bool


@dataclass
class Episode(Generic[State, Action]):
    transitions: list[Transition[State, Action]] = field(default_factory=list)

    def add(self, transition: Transition[State, Action]) -> None:
        self.transitions.append(transition)

    def __iter__(self):
        return iter(self.transitions)

    def __len__(self) -> int:
        return len(self.transitions)


class Environment(ABC, Generic[State, Action]):
    @abstractmethod
    def reset(self) -> State:
        raise NotImplementedError

    @abstractmethod
    def available_actions(self, state: State) -> list[Action]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> tuple[State, float, bool]:
        raise NotImplementedError


class Policy(ABC, Generic[State, Action]):
    @abstractmethod
    def select_action(self, state: State) -> Action:
        raise NotImplementedError


class Agent(ABC, Generic[State, Action]):
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class PredictionAgent(Agent[State, Action], ABC):
    @abstractmethod
    def update_episode(self, episode: Episode[State, Action]) -> None:
        raise NotImplementedError

    @abstractmethod
    def value_of(self, state: State) -> float:
        raise NotImplementedError
