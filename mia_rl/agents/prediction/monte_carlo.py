from __future__ import annotations

from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState


class FirstVisitMonteCarloPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def reset(self) -> None:
        self.V = defaultdict(float) #estimated value of the state
        self.N = defaultdict(int) #how many times have I already seen this state as a first visit?

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        returns = [0.0] * len(episode.transitions)
        G = 0.0
        for idx in range(len(episode.transitions) - 1, -1, -1):
            transition = episode.transitions[idx]
            G = transition.reward + self.gamma * G
            returns[idx] = G

        first_visit_index: dict[BlackjackState, int] = {}
        for idx, transition in enumerate(episode.transitions):
            first_visit_index.setdefault(transition.state, idx) 
            #if this key is not in the dictionary yet, store this value;
            #if key is already there, do nothing

        for idx, transition in enumerate(episode.transitions):
            state = transition.state
            if first_visit_index[state] != idx:
                continue
            self.N[state] += 1
            self.V[state] += (returns[idx] - self.V[state]) / self.N[state]

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])
