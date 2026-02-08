from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .env_maraffa import (
    CARD_POINTS_THIRDS,
    CARD_STRENGTH,
    CARD_SUIT,
    SUIT_MASKS,
    team_of,
)


PARAMS: tuple[float, ...] = (
    1.1958994967541685,
    0.23732132780136353,
    1.3145561114881144,
    2.9150501916851876,
    2.1061757294197907,
    2.2683148287656025,
    1.155172106006112,
    0.7173930080513861,
    4.0871172885377804,
    0.3641083368279365,
    0.47583165471112804,
    2.0887888800332526,
    8.434659265083043,
    1.0050785455262579,
    0.9273591977417727,
    0.381876215038942,
    0.7419534138672005,
    1.4954955712462221,
    1.6991269397494742,
    1.9500378843662873,
    2.259093493950632,
    -0.6452249782432833,
    2.1809659381862323,
    0.8862483106944297,
    0.9269978097802034,
    2.1438463666192784,
)


def _current_winning_team(obs: dict) -> int:
    trick_len = int(obs["trick_len"])
    if trick_len == 0:
        return -1
    trump_suit = int(obs["trump_suit"])
    lead_suit = int(obs["lead_suit"])
    trick_cards = list(obs["trick_cards"])
    trick_players = list(obs["trick_players"])

    has_trump = any(CARD_SUIT[trick_cards[i]] == trump_suit for i in range(trick_len))

    best_i = 0
    best_s = -1
    for i in range(trick_len):
        c = trick_cards[i]
        suit = CARD_SUIT[c]
        if has_trump:
            if suit != trump_suit:
                continue
        else:
            if suit != lead_suit:
                continue
        st = CARD_STRENGTH[c]
        if st > best_s:
            best_s = st
            best_i = i
    return team_of(int(trick_players[best_i]))


def _wins_if_played(obs: dict, player: int, card: int) -> bool:
    trick_len = int(obs["trick_len"])
    if trick_len == 0:
        return False

    trump_suit = int(obs["trump_suit"])
    lead_suit = int(obs["lead_suit"])
    trick_cards = list(obs["trick_cards"])
    trick_players = list(obs["trick_players"])

    cards = trick_cards[:trick_len] + [card]
    players = trick_players[:trick_len] + [player]

    has_trump = any(CARD_SUIT[c] == trump_suit for c in cards)

    best_i = 0
    best_s = -1
    for i, c in enumerate(cards):
        suit = CARD_SUIT[c]
        if has_trump:
            if suit != trump_suit:
                continue
        else:
            if suit != lead_suit:
                continue
        st = CARD_STRENGTH[c]
        if st > best_s:
            best_s = st
            best_i = i
    return int(players[best_i]) == int(player)


@dataclass
class Agent:
    name: str = "heuristic_v0"
    params: tuple[float, ...] = PARAMS

    def choose_trump(self, obs: dict, legal: List[int]) -> int:
        # legal is expected to be [0,1,2,3]
        hand = int(obs["hand_mask"])
        p = self.params
        best_s = int(legal[0])
        best_v = -1e18
        for s in legal:
            m = hand & SUIT_MASKS[s]
            cnt = m.bit_count()
            pts = 0.0
            strn = 0.0
            base = s * 10
            has3 = (hand >> (base + 0)) & 1
            has2 = (hand >> (base + 1)) & 1
            hasA = (hand >> (base + 2)) & 1
            mm = m
            while mm:
                lsb = mm & -mm
                c = lsb.bit_length() - 1
                pts += CARD_POINTS_THIRDS[c] / 3.0
                strn += CARD_STRENGTH[c] / 9.0
                mm ^= lsb
            v = p[0] * cnt + p[1] * pts + p[2] * strn + p[3] * has3 + p[4] * has2 + p[5] * hasA
            if v > best_v:
                best_v = v
                best_s = s
        return int(best_s)

    def play_card(self, obs: dict, legal: List[int]) -> int:
        if len(legal) == 1:
            return int(legal[0])
        p = self.params

        player = int(obs["player"])
        trump_suit = int(obs["trump_suit"])
        trick_len = int(obs["trick_len"])
        trick_index = int(obs["trick_index"])
        hand = int(obs["hand_mask"])

        if trick_len == 0:
            endg = 1.0 if trick_index >= 7 else 0.0
            best = int(legal[0])
            best_sc = -1e18
            for c in legal:
                tr = 1.0 if CARD_SUIT[c] == trump_suit else 0.0
                sc = p[6] * (CARD_POINTS_THIRDS[c] / 3.0) + p[7] * (CARD_STRENGTH[c] / 9.0) - p[8] * tr + p[9] * tr * endg
                suit_cnt = (hand & SUIT_MASKS[CARD_SUIT[c]]).bit_count()
                sc += p[10] * (suit_cnt / 10.0)
                if sc > best_sc:
                    best_sc = sc
                    best = c
            return int(best)

        team = team_of(player)
        partner_winning = _current_winning_team(obs) == team

        points_on_table = 0.0
        trick_cards = list(obs["trick_cards"])
        for i in range(trick_len):
            c = trick_cards[i]
            if c >= 0:
                points_on_table += CARD_POINTS_THIRDS[c] / 3.0

        winners = [c for c in legal if _wins_if_played(obs, player, c)]
        if partner_winning:
            if winners and (points_on_table >= p[11] or trick_index >= p[12]):
                return int(
                    min(
                        winners,
                        key=lambda c: (p[13] * CARD_POINTS_THIRDS[c] + p[14] * CARD_STRENGTH[c] - p[15] * points_on_table),
                    )
                )
            return int(
                min(
                    legal,
                    key=lambda c: (p[16] * CARD_POINTS_THIRDS[c] + p[17] * CARD_STRENGTH[c] + p[18] * (1 if CARD_SUIT[c] == trump_suit else 0)),
                )
            )

        if winners:
            return int(
                min(
                    winners,
                    key=lambda c: (
                        p[19] * CARD_POINTS_THIRDS[c]
                        + p[20] * CARD_STRENGTH[c]
                        + p[21] * (1 if CARD_SUIT[c] == trump_suit else 0)
                        - p[22] * points_on_table
                    ),
                )
            )
        return int(
            min(
                legal,
                key=lambda c: (p[23] * CARD_POINTS_THIRDS[c] + p[24] * CARD_STRENGTH[c] + p[25] * (1 if CARD_SUIT[c] == trump_suit else 0)),
            )
        )

