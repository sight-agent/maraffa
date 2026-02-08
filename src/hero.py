from __future__ import annotations

import copy
from dataclasses import dataclass

from .agent import Agent as HeuristicAgent
from .env_maraffa import MaraffaEnv


def _simulate_to_end(env: MaraffaEnv, agent_even, agent_odd) -> tuple[float, float]:
    """Play the rest of the hand to completion, returning (team0_pts, team1_pts)."""
    while not env.done:
        p = env.current_player
        policy = agent_even if (p & 1) == 0 else agent_odd
        if env.choose_trump_phase:
            act = policy.choose_trump(env, p)
        else:
            legal = env.legal_actions(p)
            act = policy.play_card(env, p, legal)
        env.step(act)
    return env.scores_thirds[0] / 3.0, env.scores_thirds[1] / 3.0


@dataclass
class HeroAgent:
    """A stronger agent than heuristic_v0 by doing deterministic rollouts.

    Key idea: the environment exposes full information (all hands), so we can
    evaluate candidate actions by simulating to the end assuming the opponent
    plays heuristic_v0.

    This is a best-response style policy (expensive but effective for MVP).
    """

    name: str = "hero_v1_rollout"

    def __post_init__(self):
        self._h = HeuristicAgent()
        self._opp = HeuristicAgent()

    def choose_trump(self, env: MaraffaEnv, player: int) -> int:
        # If we're not the current player, return a dummy.
        if player != env.current_player:
            return 0

        # Evaluate each trump by full rollout with both sides using heuristic after declaration.
        best_suit = 0
        best_val = -1e18
        for suit in (0, 1, 2, 3):
            e2 = copy.deepcopy(env)
            e2.step(int(suit))
            p0, p1 = _simulate_to_end(e2, self._h, self._opp)
            val = p0 - p1
            if val > best_val:
                best_val = val
                best_suit = suit
        return int(best_suit)

    def play_card(self, env: MaraffaEnv, player: int, legal: list[int]) -> int:
        if len(legal) == 1:
            return int(legal[0])

        # Rollout each legal move.
        best = int(legal[0])
        best_val = -1e18
        for c in legal:
            e2 = copy.deepcopy(env)
            e2.step(int(c))
            p0, p1 = _simulate_to_end(e2, self._h, self._opp)
            val = p0 - p1
            if val > best_val:
                best_val = val
                best = int(c)
        return int(best)
