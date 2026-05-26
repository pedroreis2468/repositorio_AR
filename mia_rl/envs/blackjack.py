from __future__ import annotations

import random
from typing import Optional

from mia_rl.core.base import Environment

BlackjackState = tuple[int, int, bool]
BlackjackAction = str

ACTIONS: tuple[BlackjackAction, ...] = ("stick", "hit")
DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] 


def draw_card(rng: random.Random) -> int:
    return rng.choice(DECK)


def draw_hand(rng: random.Random) -> list[int]:
    return [draw_card(rng), draw_card(rng)]


def usable_ace(hand: list[int]) -> bool:
    has_ace = 1 in hand     #Saves True if the hand has an ace, False otherwise.
    hand_total = sum(hand)
    can_count_ace_as_eleven = hand_total + 10 <= 21 #Saves True if counting the ace as 11 doesn't bust the hand, False otherwise.
    return has_ace and can_count_ace_as_eleven
    #can_count_ace_as_eleven is True if the hand has an ace and counting it as 11 doesn't bust the hand, False otherwise. 
    #The function returns True if both conditions are met, indicating that the hand has a usable ace, and False otherwise.


def sum_hand(hand: list[int]) -> int:
    total = sum(hand)
    if usable_ace(hand):
        return total + 10
    return total


def is_bust(hand: list[int]) -> bool:
    return sum_hand(hand) > 21


def score(hand: list[int]) -> int:
    return 0 if is_bust(hand) else sum_hand(hand)


def compare_scores(player_score: int, dealer_score: int) -> int:
    if player_score > dealer_score:
        return 1
    if player_score < dealer_score:
        return -1
    return 0


class BlackjackEnv(Environment[BlackjackState, BlackjackAction]):
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.player: list[int] = []
        self.dealer: list[int] = []

    def _state(self) -> BlackjackState:
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self) -> BlackjackState:
        self.player = draw_hand(self.rng)
        self.dealer = draw_hand(self.rng)

        while sum_hand(self.player) < 12:
            self.player.append(draw_card(self.rng))

        return self._state()

    def available_actions(self, state: BlackjackState) -> list[BlackjackAction]:
        return list(ACTIONS)

    def step(self, action: BlackjackAction) -> tuple[BlackjackState, float, bool]:
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        if action == "hit":
            self.player.append(draw_card(self.rng))
            next_state = self._state()
            if is_bust(self.player):
                return next_state, -1.0, True
            return next_state, 0.0, False

        while sum_hand(self.dealer) < 17:
            self.dealer.append(draw_card(self.rng))

        reward = float(compare_scores(score(self.player), score(self.dealer)))
        return self._state(), reward, True
