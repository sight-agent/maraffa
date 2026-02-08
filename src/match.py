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
    p = argparse.ArgumentParser(description="Play heuristic_v0 (players 0&2) vs random (players 1&3).")
    p.add_argument("--hands", type=int, default=2000)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = MaraffaEnv(seed=args.seed)
    a = Agent()
    b = RandomAgent(seed=args.seed + 1)

    wins_a = wins_b = draws = 0
    t0 = time.time()
    for i in range(int(args.hands)):
        p0, p1 = play_hand(env, a, b, seed=int(args.seed) + i)
        if p0 > p1:
            wins_a += 1
        elif p1 > p0:
            wins_b += 1
        else:
            draws += 1

    dt = time.time() - t0
    print(f"hands={args.hands} wins_record_best={wins_a} wins_random={wins_b} draws={draws}")
    print(f"winrate_record_best={(wins_a + 0.5*draws)/max(1,args.hands):.4f}")
    if dt > 0:
        print(f"hands_per_sec={args.hands/dt:.1f}")


if __name__ == "__main__":
    main()

