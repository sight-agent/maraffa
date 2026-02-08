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


def _deal_hidden_hands_with_voids(
    remaining_cards: list[int],
    fixed_hands: dict[int, int],
    needs: list[int],
    void: list[list[bool]],
    rng: random.Random,
    max_tries: int = 200,
) -> list[int] | None:
    """Assign remaining_cards to non-fixed players respecting void suits.

    fixed_hands contains already-known hands (e.g. both teammates). Those cards
    must not appear in the dealt hidden hands.

    This avoids heavy rejection by dealing only from allowed card pools.
    """
    fixed_hands = {int(p): int(h) for p, h in fixed_hands.items()}

    # Pre-split cards by suit for faster allowed checks.
    by_suit = [[], [], [], []]
    for c in remaining_cards:
        by_suit[CARD_SUIT[c]].append(c)

    for _ in range(max_tries):
        # Fresh mutable pools.
        pools = [lst[:] for lst in by_suit]
        for s in range(4):
            rng.shuffle(pools[s])

        hands = [0, 0, 0, 0]
        for p, h in fixed_hands.items():
            hands[p] = h

        # Deal to players with the tightest constraints first.
        players = [p for p in range(4) if p not in fixed_hands]
        players.sort(key=lambda p: sum(len(pools[s]) for s in range(4) if not void[p][s]))

        ok = True
        for p in players:
            need = int(needs[p])
            m = 0
            for _k in range(need):
                allowed_suits = [s for s in range(4) if (not void[p][s]) and pools[s]]
                if not allowed_suits:
                    ok = False
                    break
                s = max(allowed_suits, key=lambda ss: len(pools[ss]))
                c = pools[s].pop()
                m |= 1 << c
            if not ok:
                break
            hands[p] = m

        if not ok:
            continue

        # Sanity: total cards assigned equals expected.
        tot = 0
        for p in range(4):
            tot += hands[p].bit_count()
        expected = len(remaining_cards) + sum(h.bit_count() for h in fixed_hands.values())
        if tot != expected:
            continue

        return hands

    return None


def _determinize_env(
    obs: dict,
    rng: random.Random,
    max_tries: int = 200,
    known_hands: dict[int, int] | None = None,
) -> MaraffaEnv:
    """Create a fully-specified env consistent with the (masked) observation."""

    me = int(obs["player"])
    my_hand = int(obs["hand_mask"])
    played_mask = int(obs["played_mask"])
    void = _void_suits_from_history(obs)

    known_hands = dict(known_hands or {})
    known_hands[me] = my_hand

    # Cards that are already known to be not in any hand.
    unavailable = played_mask
    for h in known_hands.values():
        unavailable |= int(h)

    remaining_cards_all = [c for c in range(NUM_CARDS) if ((unavailable >> c) & 1) == 0]

    needs = [0, 0, 0, 0]
    for p in range(4):
        if p in known_hands:
            # Sanity: should match expected remaining size.
            continue
        needs[p] = _remaining_hand_size(obs, p)

    hands = _deal_hidden_hands_with_voids(
        remaining_cards_all,
        fixed_hands=known_hands,
        needs=needs,
        void=void,
        rng=rng,
        max_tries=max_tries,
    )

    if hands is None:
        # Fallback: ignore void constraints.
        remaining_cards = remaining_cards_all[:]
        rng.shuffle(remaining_cards)
        hands = [0, 0, 0, 0]
        for p, h in known_hands.items():
            hands[int(p)] = int(h)
        idx = 0
        for p in range(4):
            if p in known_hands:
                continue
            need = needs[p]
            m = 0
            for _ in range(need):
                c = remaining_cards[idx]
                idx += 1
                m |= 1 << c
            hands[p] = m

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


def _rollout_to_end(env: MaraffaEnv, policies: list[object]) -> float:
    """Roll out determinized env to terminal using policies.

    Returns team0_pts - team1_pts.
    """
    while not env.done:
        p = env.current_player
        pol = policies[p]
        legal = env.legal_actions(p)
        obs = env.obs(player=p)
        if not legal:
            # Shouldn't happen, but keep rollouts robust against inconsistent determinization.
            env.done = True
            break
        if env.choose_trump_phase:
            act = pol.choose_trump(obs, legal)
        else:
            act = pol.play_card(obs, legal)
        env.step(int(act))
    return (env.scores_thirds[0] - env.scores_thirds[1]) / 3.0


@dataclass
class HeroAgent:
    """Quality-first hero for imperfect information.

    Key upgrade vs naive sampling: this object controls BOTH players on the team
    (e.g. players 0&2 for agent_even), so it can remember both teammates' hands
    once each has been observed on their turn. This is team-level information
    sharing (not opponent hand cheating).

    Then we determinize only the opponents' hidden cards and evaluate actions by
    rollouts.

    Note: evaluation is always from team parity 0's perspective (players 0&2).
    When used as agent_odd, set team_parity=1.
    """

    name: str = "hero_v4_team_share_mc"
    team_parity: int = 0

    # Budgets (quality-first but bounded)
    trump_samples: int = 18
    samples_per_action: int = 24

    def __post_init__(self):
        self._h = HeuristicAgent()
        # Remember teammate hands within the current deal.
        self._known_team_hands: dict[int, int] = {}

    def _team_sign(self) -> float:
        return 1.0 if int(self.team_parity) == 0 else -1.0

    def _maybe_reset_memory(self, obs: dict) -> None:
        # New deal heuristic: at the very start.
        if int(obs["trick_index"]) == 0 and int(obs["trick_len"]) == 0 and int(obs["played_mask"]) == 0 and bool(obs["choose_trump_phase"]):
            self._known_team_hands = {}

    def _remember_hand(self, obs: dict) -> None:
        p = int(obs["player"])
        if (p & 1) == int(self.team_parity):
            self._known_team_hands[p] = int(obs["hand_mask"])

    def choose_trump(self, obs: dict, legal: list[int]) -> int:
        self._maybe_reset_memory(obs)
        self._remember_hand(obs)

        rng = random.Random((int(obs["hand_mask"]) ^ 0xC0FFEE) & 0xFFFFFFFF)
        best_a = int(legal[0])
        best_v = -1e18

        for a in legal:
            acc = 0.0
            for _ in range(int(self.trump_samples)):
                env = _determinize_env(obs, rng, known_hands=self._known_team_hands)
                env.step(int(a))
                # Rollout with heuristic for everyone (avoid recursive sampling inside rollouts).
                policies = [self._h, self._h, self._h, self._h]
                acc += _rollout_to_end(env, policies)
            v = self._team_sign() * (acc / float(self.trump_samples))
            if v > best_v:
                best_v = v
                best_a = int(a)
        return int(best_a)

    def play_card(self, obs: dict, legal: list[int]) -> int:
        self._maybe_reset_memory(obs)
        self._remember_hand(obs)

        if len(legal) == 1:
            return int(legal[0])

        rng = random.Random((int(obs["played_mask"]) ^ (int(obs["hand_mask"]) << 1) ^ 0xBADC0DE) & 0xFFFFFFFF)
        best_a = int(legal[0])
        best_v = -1e18

        for a in legal:
            acc = 0.0
            for _ in range(int(self.samples_per_action)):
                env = _determinize_env(obs, rng, known_hands=self._known_team_hands)
                env.step(int(a))
                # Rollout with heuristic for everyone (avoid recursive sampling inside rollouts).
                policies = [self._h, self._h, self._h, self._h]
                acc += _rollout_to_end(env, policies)
            v = self._team_sign() * (acc / float(self.samples_per_action))
            if v > best_v:
                best_v = v
                best_a = int(a)

        return int(best_a)
