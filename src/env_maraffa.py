from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Minimal, self-contained Maraffa environment (single hand).

NUM_PLAYERS = 4
NUM_SUITS = 4
NUM_RANKS = 10
NUM_CARDS = 40
NUM_TRICKS = 10

# Rank order in rules: 3 > 2 > A > K > N > J > 7 > 6 > 5 > 4
# Internal rank indices: [3,2,A,K,N,J,7,6,5,4] => 0..9
RANK_LABELS = ("3", "2", "A", "K", "N", "J", "7", "6", "5", "4")
RANK_STRENGTH = (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
RANK_POINTS_THIRDS = (1, 1, 3, 1, 1, 1, 0, 0, 0, 0)  # in thirds of a point

CARD_SUIT = [c // NUM_RANKS for c in range(NUM_CARDS)]
CARD_RANK = [c % NUM_RANKS for c in range(NUM_CARDS)]
CARD_STRENGTH = [RANK_STRENGTH[CARD_RANK[c]] for c in range(NUM_CARDS)]
CARD_POINTS_THIRDS = [RANK_POINTS_THIRDS[CARD_RANK[c]] for c in range(NUM_CARDS)]

SUIT_MASKS = []
for s in range(NUM_SUITS):
    m = 0
    for r in range(NUM_RANKS):
        m |= 1 << (s * NUM_RANKS + r)
    SUIT_MASKS.append(m)
SUIT_MASKS = tuple(SUIT_MASKS)

MARAFFA_MASKS = []
for s in range(NUM_SUITS):
    base = s * NUM_RANKS
    # ranks 3,2,A are indices 0,1,2 in this encoding
    MARAFFA_MASKS.append((1 << base) | (1 << (base + 1)) | (1 << (base + 2)))
MARAFFA_MASKS = tuple(MARAFFA_MASKS)


def team_of(player: int) -> int:
    return player & 1


def iter_cards(mask: int):
    m = int(mask)
    while m:
        lsb = m & -m
        c = lsb.bit_length() - 1
        yield c
        m ^= lsb


@dataclass
class TrickSnapshot:
    cards: Tuple[int, int, int, int]
    players: Tuple[int, int, int, int]
    trick_len: int
    lead_suit: int


class MaraffaEnv:
    """Single-hand Maraffa environment.

    Phase 1: declarer chooses trump (action in [0..3]).
    Phase 2: normal play (action is card id in [0..39]).
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._deck = list(range(NUM_CARDS))

        self.hands = [0, 0, 0, 0]
        self.current_player = 0
        self.declarer = 0
        self.choose_trump_phase = True
        self.trump_suit = -1

        self.trick_cards = [-1, -1, -1, -1]
        self.trick_players = [-1, -1, -1, -1]
        self.trick_len = 0
        self.lead_suit = -1
        self.trick_index = 0

        self.scores_thirds = [0, 0]
        self.bonus_team = -1
        self.played_mask = 0
        self.trick_history: List[Tuple[int, Tuple[int, int, int, int], Tuple[int, int, int, int]]] = []
        self.done = False

    def reset(self, seed: int | None = None) -> Dict[str, object]:
        if seed is not None:
            self.rng.seed(seed)

        deck = self._deck
        self.rng.shuffle(deck)
        self.hands = [0, 0, 0, 0]
        for i, c in enumerate(deck):
            self.hands[i & 3] |= 1 << c

        self.declarer = self.rng.randrange(NUM_PLAYERS)
        self.current_player = self.declarer
        self.choose_trump_phase = True
        self.trump_suit = -1

        self.trick_cards = [-1, -1, -1, -1]
        self.trick_players = [-1, -1, -1, -1]
        self.trick_len = 0
        self.lead_suit = -1
        self.trick_index = 0

        self.scores_thirds = [0, 0]
        self.bonus_team = -1
        self.played_mask = 0
        self.trick_history = []
        self.done = False
        return self.obs()

    def obs(self, player: int | None = None) -> Dict[str, object]:
        """Return a player-centric observation.

        If player is provided, the observation includes ONLY that player's hand
        (hand_mask) and omits other players' private information.

        Note: this env is still perfect-information internally, but we enforce
        fairness by never passing the env object itself to agents.
        """

        base = {
            "current_player": int(self.current_player),
            "declarer": int(self.declarer),
            "choose_trump_phase": bool(self.choose_trump_phase),
            "trump_suit": int(self.trump_suit),
            "trick_cards": tuple(int(x) for x in self.trick_cards),
            "trick_players": tuple(int(x) for x in self.trick_players),
            "trick_len": int(self.trick_len),
            "lead_suit": int(self.lead_suit),
            "trick_index": int(self.trick_index),
            "scores_thirds": tuple(int(x) for x in self.scores_thirds),
            "bonus_team": int(self.bonus_team),
            "played_mask": int(self.played_mask),
            "trick_history": self.trick_history[:],  # public (cards already played)
            "done": bool(self.done),
        }
        if player is None:
            # Internal/debug only (not for agents).
            base["hands"] = self.hands[:]
            return base

        base["player"] = int(player)
        base["hand_mask"] = int(self.hands[player])
        return base

    def legal_actions(self, player: int) -> List[int]:
        if self.choose_trump_phase:
            if player != self.current_player:
                return []
            return [0, 1, 2, 3]

        hand = self.hands[player]
        if self.trick_len == 0:
            return list(iter_cards(hand))
        led = hand & SUIT_MASKS[self.lead_suit]
        if led:
            return list(iter_cards(led))
        return list(iter_cards(hand))

    def step(self, action: int) -> Dict[str, object]:
        if self.done:
            return self.obs()
        if self.choose_trump_phase:
            self._declare_trump(int(action))
            self.current_player = self.declarer
            return self.obs()
        self._play_card(self.current_player, int(action))
        return self.obs()

    def _declare_trump(self, suit: int) -> None:
        self.trump_suit = suit
        self.choose_trump_phase = False
        maraffa_mask = MARAFFA_MASKS[suit]
        t0 = self.hands[0] | self.hands[2]
        t1 = self.hands[1] | self.hands[3]
        if (t0 & maraffa_mask) == maraffa_mask:
            self.bonus_team = 0
        elif (t1 & maraffa_mask) == maraffa_mask:
            self.bonus_team = 1
        else:
            self.bonus_team = -1

    def _play_card(self, player: int, card: int) -> None:
        self.hands[player] ^= 1 << card
        self.played_mask |= 1 << card
        pos = self.trick_len
        self.trick_cards[pos] = card
        self.trick_players[pos] = player
        self.trick_len += 1
        if pos == 0:
            self.lead_suit = CARD_SUIT[card]

        if self.trick_len < NUM_PLAYERS:
            self.current_player = (player + 1) & 3
            return

        winner, thirds = self._resolve_trick()
        self.trick_history.append((self.lead_suit, tuple(self.trick_players), tuple(self.trick_cards)))
        self.scores_thirds[team_of(winner)] += thirds
        self.trick_index += 1

        self.trick_cards = [-1, -1, -1, -1]
        self.trick_players = [-1, -1, -1, -1]
        self.trick_len = 0
        self.lead_suit = -1
        self.current_player = winner

        if self.trick_index == NUM_TRICKS:
            self.done = True
            if self.bonus_team in (0, 1):
                self.scores_thirds[self.bonus_team] += 9

    def _resolve_trick(self) -> Tuple[int, int]:
        has_trump = False
        for c in self.trick_cards:
            if CARD_SUIT[c] == self.trump_suit:
                has_trump = True
                break
        best_i = 0
        best_s = -1
        for i, c in enumerate(self.trick_cards):
            suit = CARD_SUIT[c]
            if has_trump:
                if suit != self.trump_suit:
                    continue
            else:
                if suit != self.lead_suit:
                    continue
            st = CARD_STRENGTH[c]
            if st > best_s:
                best_s = st
                best_i = i
        winner = self.trick_players[best_i]
        thirds = 0
        for c in self.trick_cards:
            thirds += CARD_POINTS_THIRDS[c]
        return int(winner), int(thirds)

