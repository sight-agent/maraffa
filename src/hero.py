from __future__ import annotations

from dataclasses import dataclass

from .env_maraffa import (
    CARD_POINTS_THIRDS,
    CARD_STRENGTH,
    CARD_SUIT,
    MARAFFA_MASKS,
    SUIT_MASKS,
    MaraffaEnv,
    team_of,
)


def _current_winner_team(obs: dict) -> int:
    tl = int(obs["trick_len"])
    if tl == 0:
        return -1
    trump = int(obs["trump_suit"])
    lead = int(obs["lead_suit"])
    cards = list(obs["trick_cards"])
    players = list(obs["trick_players"])

    has_trump = any(CARD_SUIT[cards[i]] == trump for i in range(tl))
    best_i = 0
    best_s = -1
    for i in range(tl):
        c = cards[i]
        suit = CARD_SUIT[c]
        if has_trump:
            if suit != trump:
                continue
        else:
            if suit != lead:
                continue
        st = CARD_STRENGTH[c]
        if st > best_s:
            best_s = st
            best_i = i
    return team_of(int(players[best_i]))


def _wins_if_played(obs: dict, player: int, card: int) -> bool:
    tl = int(obs["trick_len"])
    if tl == 0:
        return False
    trump = int(obs["trump_suit"])
    lead = int(obs["lead_suit"])
    cards = list(obs["trick_cards"])
    players = list(obs["trick_players"])

    cards2 = cards[:tl] + [card]
    players2 = players[:tl] + [player]

    has_trump = any(CARD_SUIT[c] == trump for c in cards2)
    best_i = 0
    best_s = -1
    for i, c in enumerate(cards2):
        suit = CARD_SUIT[c]
        if has_trump:
            if suit != trump:
                continue
        else:
            if suit != lead:
                continue
        st = CARD_STRENGTH[c]
        if st > best_s:
            best_s = st
            best_i = i
    return int(players2[best_i]) == int(player)


def _public_void_suits(obs: dict) -> list[list[bool]]:
    """Infer void suits for each player from public history (including current trick)."""
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

    tl = int(obs["trick_len"])
    if tl > 0:
        ls = int(obs["lead_suit"])
        players = list(obs["trick_players"])
        cards = list(obs["trick_cards"])
        for i in range(tl):
            p = int(players[i])
            c = int(cards[i])
            if c >= 0 and CARD_SUIT[c] != ls:
                void[p][ls] = True
    return void


def _seen_cards_mask(obs: dict) -> int:
    """All publicly seen cards (played so far + current trick)."""
    m = int(obs["played_mask"])
    for c in list(obs["trick_cards"]):
        if int(c) >= 0:
            m |= 1 << int(c)
    return m


def _high_cards_seen_fraction(obs: dict, suit: int) -> float:
    """How many of the top 3 ranks (3,2,A) of suit have been seen."""
    seen = _seen_cards_mask(obs)
    base = suit * 10
    top = [base + 0, base + 1, base + 2]
    return sum(1 for c in top if (seen >> c) & 1) / 3.0


@dataclass
class HeroAgent:
    """heuristic_v1: heuristic-style agent (no search, no cheating).

    Upgrades over heuristic_v0:
    - Uses public trick_history to infer void suits.
    - Uses public seen cards to estimate suit safety.
    - Slightly better trump choice using void+safety signals.
    """

    name: str = "heuristic_v1"

    # weights (tweakable)
    w_cnt: float = 1.20
    w_pts: float = 0.25
    w_str: float = 1.30
    w_maraffa: float = 2.60

    w_void_bonus: float = 0.25
    w_seen_high: float = 0.30

    # play weights
    w_lead_pts: float = 1.10
    w_lead_str: float = 0.70
    w_lead_trump_penalty: float = 3.50
    w_lead_suit_len: float = 0.60
    w_lead_seen_high: float = 0.80

    w_follow_win_low: float = 1.0
    w_follow_win_take_pts: float = 1.3
    w_follow_dump_low: float = 1.0
    w_follow_dump_trump_penalty: float = 1.2

    def choose_trump(self, obs: dict, legal: list[int]) -> int:
        hand = int(obs["hand_mask"])
        void = _public_void_suits(obs)
        me = int(obs["player"])
        partner = me ^ 2

        best_s = int(legal[0])
        best_v = -1e18
        for s in legal:
            m = hand & SUIT_MASKS[s]
            cnt = m.bit_count()

            pts = 0.0
            strn = 0.0
            mm = m
            while mm:
                lsb = mm & -mm
                c = lsb.bit_length() - 1
                pts += CARD_POINTS_THIRDS[c] / 3.0
                strn += CARD_STRENGTH[c] / 9.0
                mm ^= lsb

            has_maraffa = 1.0 if (hand & MARAFFA_MASKS[s]) == MARAFFA_MASKS[s] else 0.0

            # public features
            void_bonus = 0.0
            # if next opponents are void in suit, leading it later is safer
            for opp in ((me + 1) & 3, (me + 3) & 3):
                if void[opp][s]:
                    void_bonus += 1.0
            # if partner void, trumping later might be good, but also coordination is hard
            if void[partner][s]:
                void_bonus += 0.25

            seen_high = _high_cards_seen_fraction(obs, s)

            v = (
                self.w_cnt * cnt
                + self.w_pts * pts
                + self.w_str * strn
                + self.w_maraffa * has_maraffa
                + self.w_void_bonus * void_bonus
                + self.w_seen_high * seen_high
            )
            if v > best_v:
                best_v = v
                best_s = int(s)
        return int(best_s)

    def play_card(self, obs: dict, legal: list[int]) -> int:
        if len(legal) == 1:
            return int(legal[0])

        me = int(obs["player"])
        trump = int(obs["trump_suit"])
        trick_len = int(obs["trick_len"])
        trick_index = int(obs["trick_index"])

        # Lead: prefer long safe suits; avoid wasting trump early; use seen-high as safety.
        if trick_len == 0:
            hand = int(obs["hand_mask"])
            endg = 1.0 if trick_index >= 7 else 0.0
            best = int(legal[0])
            best_sc = -1e18
            for c in legal:
                s = CARD_SUIT[c]
                tr = 1.0 if s == trump else 0.0
                suit_cnt = (hand & SUIT_MASKS[s]).bit_count()
                seen_high = _high_cards_seen_fraction(obs, s)

                sc = 0.0
                sc += self.w_lead_pts * (CARD_POINTS_THIRDS[c] / 3.0)
                sc += self.w_lead_str * (CARD_STRENGTH[c] / 9.0)
                sc -= self.w_lead_trump_penalty * tr * (1.0 - endg)
                sc += self.w_lead_suit_len * (suit_cnt / 10.0)
                sc += self.w_lead_seen_high * seen_high

                if sc > best_sc:
                    best_sc = sc
                    best = int(c)
            return int(best)

        # Following: similar structure to heuristic_v0 but with a slightly more
        # consistent points model.
        team = team_of(me)
        partner_winning = _current_winner_team(obs) == team

        points_on_table = 0.0
        cards = list(obs["trick_cards"])
        for i in range(trick_len):
            c = int(cards[i])
            if c >= 0:
                points_on_table += CARD_POINTS_THIRDS[c] / 3.0

        winners = [c for c in legal if _wins_if_played(obs, me, c)]

        if partner_winning:
            # If partner is winning, dump low value; only take if large points at risk.
            if winners and points_on_table >= 1.5:
                return int(
                    min(
                        winners,
                        key=lambda c: (
                            self.w_follow_win_take_pts * CARD_POINTS_THIRDS[c]
                            + self.w_follow_win_low * CARD_STRENGTH[c]
                            - 0.8 * points_on_table
                        ),
                    )
                )
            return int(
                min(
                    legal,
                    key=lambda c: (
                        self.w_follow_dump_low * CARD_POINTS_THIRDS[c]
                        + 0.9 * CARD_STRENGTH[c]
                        + self.w_follow_dump_trump_penalty * (1 if CARD_SUIT[c] == trump else 0)
                    ),
                )
            )

        if winners:
            # Take with the cheapest winning card, but prioritize capturing points.
            return int(
                min(
                    winners,
                    key=lambda c: (
                        1.0 * CARD_POINTS_THIRDS[c]
                        + 1.2 * CARD_STRENGTH[c]
                        + 1.1 * (1 if CARD_SUIT[c] == trump else 0)
                        - 1.0 * points_on_table
                    ),
                )
            )

        # Cannot win: dump low.
        return int(
            min(
                legal,
                key=lambda c: (
                    1.0 * CARD_POINTS_THIRDS[c]
                    + 1.0 * CARD_STRENGTH[c]
                    + 1.0 * (1 if CARD_SUIT[c] == trump else 0)
                ),
            )
        )
