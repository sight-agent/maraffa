from __future__ import annotations

import random
from dataclasses import dataclass

from .agent import Agent as HeuristicAgent
from .env_maraffa import MaraffaEnv, CARD_SUIT, NUM_CARDS


def _player_cards_played_so_far(obs: dict, p: int) -> int:
    """How many cards player p has already played (completed tricks + current trick)."""
    trick_index = int(obs["trick_index"])
    trick_len = int(obs["trick_len"])
    trick_players = list(obs["trick_players"])
    in_current = 1 if p in trick_players[:trick_len] else 0
    return trick_index + in_current


def _remaining_hand_size(obs: dict, p: int) -> int:
    return 10 - _player_cards_played_so_far(obs, p)


def _void_suits_from_history(obs: dict) -> list[list[bool]]:
    """Infer suits that each player is certainly void in (based on failing to follow suit)."""
    void = [[False] * 4 for _ in range(4)]
    for lead_suit, players, cards in obs.get("trick_history", []):
        ls = int(lead_suit)
        for p, c in zip(players, cards):
            p = int(p)
            c = int(c)
            if c < 0:
                continue
            if CARD_SUIT[c] != ls:
                void[p][ls] = True
    # Also consider current trick so far.
    if int(obs["trick_len"]) > 0:
        ls = int(obs["lead_suit"])
        trick_players = list(obs["trick_players"])
        trick_cards = list(obs["trick_cards"])
        for i in range(int(obs["trick_len"])):
            p = int(trick_players[i])
            c = int(trick_cards[i])
            if c >= 0 and CARD_SUIT[c] != ls:
                void[p][ls] = True
    return void


def _determinize_env(obs: dict, rng: random.Random, max_tries: int = 200) -> MaraffaEnv:
    """Create a fully-specified env consistent with the (masked) observation.

    We only know our own hand; we sample the unknown hands uniformly from the
    remaining cards, respecting each player's remaining hand size.

    We also enforce a lightweight public-information constraint: if a player
    failed to follow suit in the public history, we do not give them any cards
    of that suit in their current (unplayed) hand.
    """

    me = int(obs["player"])
    my_hand = int(obs["hand_mask"])
    played_mask = int(obs["played_mask"])
    void = _void_suits_from_history(obs)

    # Cards that are already known to be not in any hand.
    unavailable = my_hand | played_mask
    remaining_cards_all = [c for c in range(NUM_CARDS) if ((unavailable >> c) & 1) == 0]

    for _attempt in range(max_tries):
        remaining_cards = remaining_cards_all[:]
        rng.shuffle(remaining_cards)

        hands = [0, 0, 0, 0]
        hands[me] = my_hand

        idx = 0
        ok = True
        for p in range(4):
            if p == me:
                continue
            need = _remaining_hand_size(obs, p)
            m = 0
            for _ in range(need):
                if idx >= len(remaining_cards):
                    ok = False
                    break
                c = remaining_cards[idx]
                idx += 1
                m |= 1 << c
            if not ok:
                break
            # Enforce void suit constraints on the CURRENT hand.
            for s in range(4):
                if void[p][s]:
                    # if any card in suit s is present, reject.
                    # suit cards are contiguous blocks of 10.
                    base = s * 10
                    suit_mask = ((1 << 10) - 1) << base
                    if m & suit_mask:
                        ok = False
                        break
            if not ok:
                break
            hands[p] = m

        if not ok:
            continue

        env = MaraffaEnv(seed=0)
        env.hands = hands
        env.current_player = int(obs["current_player"])
        env.declarer = int(obs["declarer"])
        env.choose_trump_phase = bool(obs["choose_trump_phase"])
        env.trump_suit = int(obs["trump_suit"])

        tc = list(obs["trick_cards"])
        tp = list(obs["trick_players"])
        env.trick_cards = [int(x) for x in tc]
        env.trick_players = [int(x) for x in tp]
        env.trick_len = int(obs["trick_len"])
        env.lead_suit = int(obs["lead_suit"])
        env.trick_index = int(obs["trick_index"])

        st = list(obs["scores_thirds"])
        env.scores_thirds = [int(st[0]), int(st[1])]
        env.bonus_team = int(obs["bonus_team"])
        env.played_mask = int(obs["played_mask"])
        env.trick_history = obs.get("trick_history", [])[:]
        env.done = bool(obs["done"])

        return env

    # Fallback: if constraints are too tight, ignore them.
    rng.shuffle(remaining_cards_all)
    hands = [0, 0, 0, 0]
    hands[me] = my_hand
    idx = 0
    for p in range(4):
        if p == me:
            continue
        need = _remaining_hand_size(obs, p)
        m = 0
        for _ in range(need):
            c = remaining_cards_all[idx]
            idx += 1
            m |= 1 << c
        hands[p] = m

    env = MaraffaEnv(seed=0)
    env.hands = hands
    env.current_player = int(obs["current_player"])
    env.declarer = int(obs["declarer"])
    env.choose_trump_phase = bool(obs["choose_trump_phase"])
    env.trump_suit = int(obs["trump_suit"])
    env.trick_cards = [int(x) for x in list(obs["trick_cards"])]
    env.trick_players = [int(x) for x in list(obs["trick_players"])]
    env.trick_len = int(obs["trick_len"])
    env.lead_suit = int(obs["lead_suit"])
    env.trick_index = int(obs["trick_index"])
    st = list(obs["scores_thirds"])
    env.scores_thirds = [int(st[0]), int(st[1])]
    env.bonus_team = int(obs["bonus_team"])
    env.played_mask = int(obs["played_mask"])
    env.trick_history = obs.get("trick_history", [])[:]
    env.done = bool(obs["done"])
    return env


def _rollout_to_end(env: MaraffaEnv, policies: list[object]) -> float:
    """Roll out determinized env to terminal using policies.

    Returns team0_pts - team1_pts.
    """
    while not env.done:
        p = env.current_player
        pol = policies[p]
        legal = env.legal_actions(p)
        obs = env.obs(player=p)
        if env.choose_trump_phase:
            act = pol.choose_trump(obs, legal)
        else:
            act = pol.play_card(obs, legal)
        env.step(int(act))
    return (env.scores_thirds[0] - env.scores_thirds[1]) / 3.0


@dataclass
class HeroAgent:
    """Non-cheating hero agent (imperfect information).

    Uses determinization (sampling unknown hands) + rollouts with heuristic_v0.

    This is a basic ISMCTS-style approach: for each candidate action, sample N
    consistent hidden states, apply the action, then rollout to end.
    """

    name: str = "hero_v2_determinization"
    samples_per_action: int = 12

    def __post_init__(self):
        self._h = HeuristicAgent()

    def choose_trump(self, obs: dict, legal: list[int]) -> int:
        # Evaluate each trump suit by sampling.
        rng = random.Random((int(obs["played_mask"]) ^ (int(obs["hand_mask"]) << 1) ^ 0xC0FFEE) & 0xFFFFFFFF)
        best_a = int(legal[0])
        best_v = -1e18

        for a in legal:
            acc = 0.0
            for _ in range(self.samples_per_action):
                env = _determinize_env(obs, rng)
                env.step(int(a))
                # Everyone uses heuristic during rollout (including us after first move).
                policies = [self._h, self._h, self._h, self._h]
                acc += _rollout_to_end(env, policies)
            v = acc / float(self.samples_per_action)
            if v > best_v:
                best_v = v
                best_a = int(a)
        return int(best_a)

    def play_card(self, obs: dict, legal: list[int]) -> int:
        if len(legal) == 1:
            return int(legal[0])

        rng = random.Random((int(obs["played_mask"]) ^ (int(obs["hand_mask"]) << 1) ^ 0xBADC0DE) & 0xFFFFFFFF)
        best_a = int(legal[0])
        best_v = -1e18

        # To keep it fast, sample a fixed number per action.
        for a in legal:
            acc = 0.0
            for _ in range(self.samples_per_action):
                env = _determinize_env(obs, rng)
                env.step(int(a))
                policies = [self._h, self._h, self._h, self._h]
                acc += _rollout_to_end(env, policies)
            v = acc / float(self.samples_per_action)
            if v > best_v:
                best_v = v
                best_a = int(a)
        return int(best_a)
