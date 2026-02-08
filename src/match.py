from __future__ import annotations

import argparse
import random
import time

from .agent import Agent
from .env_maraffa import MaraffaEnv


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose_trump(self, obs: dict, legal: list[int]) -> int:
        return int(self.rng.choice(legal))

    def play_card(self, obs: dict, legal: list[int]) -> int:
        return int(self.rng.choice(legal))


def play_hand(env: MaraffaEnv, agent_even, agent_odd, seed: int) -> tuple[float, float]:
    env.reset(seed=seed)
    return play_hand_from_state(env, agent_even, agent_odd)


def play_hand_from_state(env: MaraffaEnv, agent_even, agent_odd) -> tuple[float, float]:
    """Play from current env state (already reset)."""
    while not env.done:
        p = env.current_player
        policy = agent_even if (p & 1) == 0 else agent_odd
        legal = env.legal_actions(p)
        obs = env.obs(player=p)
        if env.choose_trump_phase:
            act = policy.choose_trump(obs, legal)
        else:
            act = policy.play_card(obs, legal)
        env.step(act)
    # Convert thirds to points for reporting
    return env.scores_thirds[0] / 3.0, env.scores_thirds[1] / 3.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Play heuristic_v0 (players 0&2) vs random (players 1&3). "
            "Runs paired tests (2 hands per deal with a 1-seat rotation) and reports pentanomial."
        )
    )
    p.add_argument("--hands", type=int, default=2000, help="Total hands (will be rounded down to an even number).")
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def _rotate_hands_by_1(hands: list[int]) -> list[int]:
    # New player p gets the hand that was previously at (p-1)
    return [int(hands[(p - 1) & 3]) for p in range(4)]


def _score_of_result(p0: float, p1: float) -> float:
    if p0 > p1:
        return 1.0
    if p1 > p0:
        return 0.0
    return 0.5


def main() -> None:
    args = parse_args()
    env = MaraffaEnv(seed=args.seed)
    a = Agent()
    b = RandomAgent(seed=args.seed + 1)

    # Paired test: 2 hands per deal.
    total_hands = int(args.hands)
    total_hands -= total_hands % 2
    deals = total_hands // 2

    wins_a = wins_b = draws = 0
    # Pentanomial bins by total score over the 2 paired games: 2, 1.5, 1, 0.5, 0
    penta = {2.0: 0, 1.5: 0, 1.0: 0, 0.5: 0, 0.0: 0}

    t0 = time.time()
    for i in range(deals):
        seed = int(args.seed) + i
        hands, declarer = MaraffaEnv.deal_from_seed(seed)

        # Game 1: as dealt.
        env.reset_from(hands, declarer)
        p0, p1 = play_hand_from_state(env, a, b)
        s1 = _score_of_result(p0, p1)
        if p0 > p1:
            wins_a += 1
        elif p1 > p0:
            wins_b += 1
        else:
            draws += 1

        # Game 2: same deal, but hands rotated by 1 seat (swaps team parity fairness).
        hands2 = _rotate_hands_by_1(hands)
        declarer2 = (declarer + 1) & 3
        env.reset_from(hands2, declarer2)
        p0b, p1b = play_hand_from_state(env, a, b)
        s2 = _score_of_result(p0b, p1b)
        if p0b > p1b:
            wins_a += 1
        elif p1b > p0b:
            wins_b += 1
        else:
            draws += 1

        penta[s1 + s2] += 1

    dt = time.time() - t0
    print(f"hands={total_hands} deals={deals}")
    print(f"W/D/L={wins_a}/{draws}/{wins_b}")
    print(f"winrate_record_best={(wins_a + 0.5*draws)/max(1,total_hands):.4f}")
    print("pentanomial (per deal, 2 hands/deal):")
    print(f"  2.0  (WW): {penta[2.0]}")
    print(f"  1.5 (W+D): {penta[1.5]}")
    print(f"  1.0 (WL/DD): {penta[1.0]}")
    print(f"  0.5 (L+D): {penta[0.5]}")
    print(f"  0.0  (LL): {penta[0.0]}")
    if dt > 0:
        print(f"hands_per_sec={total_hands/dt:.1f}")


if __name__ == "__main__":
    main()

