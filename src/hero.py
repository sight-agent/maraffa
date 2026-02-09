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
        c = int(cards[i])
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

    cards2 = [int(x) for x in cards[:tl]] + [int(card)]
    players2 = [int(x) for x in players[:tl]] + [int(player)]

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
        c = int(c)
        if c >= 0:
            m |= 1 << c
    return m


def _suit_seen_count(obs: dict, suit: int) -> int:
    seen = _seen_cards_mask(obs)
    base = suit * 10
    suit_mask = ((1 << 10) - 1) << base
    return int(((seen & suit_mask) >> base).bit_count())


def _high_cards_seen_fraction(obs: dict, suit: int) -> float:
    """How many of the top 3 ranks (3,2,A) of suit have been seen."""
    seen = _seen_cards_mask(obs)
    base = suit * 10
    top = [base + 0, base + 1, base + 2]
    return sum(1 for c in top if (seen >> c) & 1) / 3.0


def _void_risk_for_lead(obs: dict, suit: int) -> float:
    """Risk that leading suit will be trump-cut by opponents.

    Very simple model: count how many opponents are known void in the suit.
    """
    void = _public_void_suits(obs)
    me = int(obs["player"])
    opps = [(me + 1) & 3, (me + 3) & 3]
    return float(sum(1 for o in opps if void[o][suit]))


def _build_determinized_env_from_obs(obs: dict, rng, *, max_tries: int = 80) -> MaraffaEnv:
    """Create a full env consistent with obs by sampling hidden opponent hands.

    Constraints enforced:
    - current player's hand is fixed to obs['hand_mask']
    - cards in played_mask + current trick are removed from the deck
    - void suits inferred from public history are respected (no cards of that suit)

    This is used only internally for lightweight Monte Carlo (not exposed to agents).
    """

    me = int(obs["player"])
    my_hand = int(obs["hand_mask"])

    # Build void constraints.
    void = _public_void_suits(obs)

    # Cards already out of the game (played + current trick) and my hand.
    unavailable = int(obs["played_mask"]) | int(my_hand)
    for c in list(obs["trick_cards"]):
        c = int(c)
        if c >= 0:
            unavailable |= 1 << c

    remaining = [c for c in range(40) if ((unavailable >> c) & 1) == 0]

    # Remaining hand sizes for others.
    trick_index = int(obs["trick_index"])
    trick_players = list(obs["trick_players"])
    trick_len = int(obs["trick_len"])

    def played_by(p: int) -> int:
        # completed tricks + already played in current trick
        in_cur = 1 if p in [int(x) for x in trick_players[:trick_len]] else 0
        return trick_index + in_cur

    need = [0, 0, 0, 0]
    for p in range(4):
        if p == me:
            continue
        need[p] = 10 - played_by(p)

    # Prepare suit pools.
    by_suit = [[], [], [], []]
    for c in remaining:
        by_suit[CARD_SUIT[c]].append(c)

    for _ in range(max_tries):
        pools = [lst[:] for lst in by_suit]
        for s in range(4):
            rng.shuffle(pools[s])

        hands = [0, 0, 0, 0]
        hands[me] = my_hand

        # Deal to others; void constraints applied.
        ok = True
        for p in range(4):
            if p == me:
                continue
            m = 0
            for _k in range(need[p]):
                allowed = [s for s in range(4) if (not void[p][s]) and pools[s]]
                if not allowed:
                    ok = False
                    break
                s = max(allowed, key=lambda ss: len(pools[ss]))
                c = pools[s].pop()
                m |= 1 << c
            if not ok:
                break
            hands[p] = m

        if not ok:
            continue

        env = MaraffaEnv(seed=0)
        env.hands = [int(x) for x in hands]
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

    # Fallback: ignore voids.
    rng.shuffle(remaining)
    hands = [0, 0, 0, 0]
    hands[me] = my_hand
    idx = 0
    for p in range(4):
        if p == me:
            continue
        m = 0
        for _k in range(need[p]):
            c = remaining[idx]
            idx += 1
            m |= 1 << c
        hands[p] = m

    env = MaraffaEnv(seed=0)
    env.hands = [int(x) for x in hands]
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


@dataclass
class HeroAgent:
    """heuristic_v3_hybrid: heuristic + small Monte Carlo on critical moves.

    Still no cheating:
    - uses only own hand + public info
    - no teammate info sharing

    On some decisions (especially when following and points are on table), it
    samples plausible hidden hands and evaluates candidate actions by short
    rollouts with heuristic_v0.
    """

    name: str = "heuristic_v3_hybrid"

    # --- Monte Carlo settings (keep small: VPS-friendly)
    mc_enabled: bool = True
    mc_samples: int = 6
    mc_rollout_tricks: int = 1  # finish current trick; 2 => also next trick

    # --- Trump choice weights
    w_cnt: float = 1.20
    w_pts: float = 0.25
    w_str: float = 1.30
    w_maraffa: float = 2.60
    w_void_bonus: float = 0.25
    w_seen_high: float = 0.30

    # --- Lead weights
    w_lead_pts: float = 1.10
    w_lead_str: float = 0.70
    w_lead_trump_penalty: float = 3.50
    w_lead_suit_len: float = 0.60
    w_lead_seen_high: float = 0.80

    # new lead features
    w_lead_void_risk: float = 1.20
    w_lead_suit_seen: float = 0.25
    w_lead_draw_trump: float = 1.00

    # --- Follow weights
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

            void_bonus = 0.0
            for opp in ((me + 1) & 3, (me + 3) & 3):
                if void[opp][s]:
                    void_bonus += 1.0
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

    def _heuristic_play_card(self, obs: dict, legal: list[int]) -> int:
        """Pure heuristic (no MC) decision."""
        me = int(obs["player"])
        trump = int(obs["trump_suit"])
        trick_len = int(obs["trick_len"])
        trick_index = int(obs["trick_index"])

        # --- Lead
        if trick_len == 0:
            hand = int(obs["hand_mask"])
            endg = 1.0 if trick_index >= 7 else 0.0

            void = _public_void_suits(obs)
            opps = [(me + 1) & 3, (me + 3) & 3]
            opp_void_total = 0
            for o in opps:
                opp_void_total += sum(1 for s in range(4) if void[o][s])
            have_trump = (hand & SUIT_MASKS[trump]) != 0

            best = int(legal[0])
            best_sc = -1e18
            for c in legal:
                s = CARD_SUIT[c]
                tr = 1.0 if s == trump else 0.0
                suit_cnt = (hand & SUIT_MASKS[s]).bit_count()
                seen_high = _high_cards_seen_fraction(obs, s)
                seen_cnt = _suit_seen_count(obs, s)
                void_risk = _void_risk_for_lead(obs, s)

                sc = 0.0
                sc += self.w_lead_pts * (CARD_POINTS_THIRDS[c] / 3.0)
                sc += self.w_lead_str * (CARD_STRENGTH[c] / 9.0)
                sc -= self.w_lead_trump_penalty * tr * (1.0 - endg)
                sc += self.w_lead_suit_len * (suit_cnt / 10.0)
                sc += self.w_lead_seen_high * seen_high
                sc -= self.w_lead_void_risk * void_risk * (1.0 - endg)
                sc += self.w_lead_suit_seen * (seen_cnt / 10.0)

                if have_trump and tr > 0.0:
                    sc += self.w_lead_draw_trump * (opp_void_total / 8.0) * (1.0 - endg)

                if sc > best_sc:
                    best_sc = sc
                    best = int(c)
            return int(best)

        # --- Following
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

    def play_card(self, obs: dict, legal: list[int]) -> int:
        if len(legal) == 1:
            return int(legal[0])

        # Decide whether to run MC.
        if not self.mc_enabled:
            return int(self._heuristic_play_card(obs, legal))

        trick_len = int(obs["trick_len"])
        # MC only when following (harder decision) or when there are points on table.
        points_on_table = 0.0
        if trick_len > 0:
            cards = list(obs["trick_cards"])
            for i in range(trick_len):
                c = int(cards[i])
                if c >= 0:
                    points_on_table += CARD_POINTS_THIRDS[c] / 3.0

        do_mc = (trick_len > 0 and (points_on_table >= 1.0 or len(legal) >= 3))
        if not do_mc:
            return int(self._heuristic_play_card(obs, legal))

        import random
        from .agent import Agent as HeuristicV0

        rng = random.Random((int(obs["played_mask"]) ^ (int(obs["hand_mask"]) << 1) ^ 0xA11CE) & 0xFFFFFFFF)
        v0 = HeuristicV0()

        me = int(obs["player"])
        my_team = team_of(me)

        best_a = int(legal[0])
        best_v = -1e18

        for a in legal:
            acc = 0.0
            for _ in range(int(self.mc_samples)):
                env = _build_determinized_env_from_obs(obs, rng)
                base = (env.scores_thirds[my_team] - env.scores_thirds[1 - my_team]) / 3.0

                env.step(int(a))

                # Rollout: finish current trick and optionally next trick.
                start_trick = env.trick_index
                target_trick = start_trick + int(self.mc_rollout_tricks)
                while not env.done and env.trick_index < target_trick:
                    p = env.current_player
                    legal2 = env.legal_actions(p)
                    if not legal2:
                        env.done = True
                        break
                    obs2 = env.obs(player=p)
                    if env.choose_trump_phase:
                        act2 = v0.choose_trump(obs2, legal2)
                    else:
                        act2 = v0.play_card(obs2, legal2)
                    env.step(int(act2))

                val = (env.scores_thirds[my_team] - env.scores_thirds[1 - my_team]) / 3.0 - base
                acc += val

            v = acc / float(self.mc_samples)
            if v > best_v:
                best_v = v
                best_a = int(a)

        return int(best_a)
