from __future__ import annotations

"""Tune HeroAgent (heuristic_v1) weights against heuristic_v0.

Usage:
  python3 -m src.tune_hero --iters 200 --pop 24 --deals 200 --seed 7000

Notes:
- Uses paired testing (2 hands per deal with 1-seat rotation).
- Optimizes winrate of hero (as even team 0&2) vs heuristic_v0.
- Keeps runtime modest; evaluation is fast.
"""

import argparse
import dataclasses
import math
import random
import time
from typing import Dict, Tuple

from .agent import Agent as HeuristicV0
from .env_maraffa import MaraffaEnv
from .match import play_hand_from_state
from .hero import HeroAgent


def paired_winrate(agent_even, agent_odd, *, deals: int, seed: int) -> Tuple[float, Tuple[int, int, int]]:
    env = MaraffaEnv(seed=seed)
    w = d = l = 0

    for i in range(int(deals)):
        hands, declarer = MaraffaEnv.deal_from_seed(int(seed) + i)

        env.reset_from(hands, declarer)
        p0, p1 = play_hand_from_state(env, agent_even, agent_odd)
        if p0 > p1:
            w += 1
        elif p1 > p0:
            l += 1
        else:
            d += 1

        hands2 = [hands[(p - 1) & 3] for p in range(4)]
        env.reset_from(hands2, (declarer + 1) & 3)
        p0, p1 = play_hand_from_state(env, agent_even, agent_odd)
        if p0 > p1:
            w += 1
        elif p1 > p0:
            l += 1
        else:
            d += 1

    n = int(deals) * 2
    wr = (w + 0.5 * d) / max(1, n)
    return wr, (w, d, l)


def get_base_params() -> Dict[str, float]:
    h = HeroAgent()
    d = dataclasses.asdict(h)
    # keep only numeric weights
    keep = {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and k != "name"}
    return keep


def propose(rng: random.Random, base: Dict[str, float], sigma: float) -> Dict[str, float]:
    """Multiplicative log-normal-ish perturbation, with gentle additive noise for small weights."""
    out: Dict[str, float] = {}
    for k, v in base.items():
        v = float(v)
        # keep sane positive weights; allow a few weights to be 0..
        sign = 1.0 if v >= 0 else -1.0
        mag = abs(v)
        # multiplicative noise
        m = math.exp(rng.gauss(0.0, sigma))
        cand = sign * mag * m
        # small additive for tiny mags
        cand += rng.gauss(0.0, 0.05 * sigma)
        # clip
        if k.startswith("w_"):
            cand = max(-20.0, min(20.0, cand))
        out[k] = float(cand)

    # A couple of practical constraints
    out["w_lead_trump_penalty"] = max(0.0, out["w_lead_trump_penalty"])
    out["w_follow_dump_trump_penalty"] = max(0.0, out["w_follow_dump_trump_penalty"])
    out["w_maraffa"] = max(0.0, out["w_maraffa"])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--pop", type=int, default=12, help="Number of mutated candidates per iteration (plus the current best).")
    p.add_argument("--deals", type=int, default=2000, help="Paired deals per evaluation (hands = deals*2).")
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=30, help="Stop after this many iterations without improvement.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    base = get_base_params()

    v0 = HeuristicV0()

    # Train/val are two different deal streams. To reduce overfitting to a fixed seed,
    # we increment the seed at every iteration (but keep it identical across candidates
    # within the same iteration for a fair comparison).

    best = dict(base)
    sigma = float(args.sigma)

    # Initial baseline (iteration 0 seeds)
    seed0_train = int(args.seed) + 123
    seed0_val = int(args.seed) + 999
    best_train, _ = paired_winrate(HeroAgent(**best), v0, deals=args.deals, seed=seed0_train)
    best_val, _ = paired_winrate(HeroAgent(**best), v0, deals=args.deals, seed=seed0_val)

    print(f"base train_wr={best_train:.4f} val_wr={best_val:.4f} sigma={sigma:.3f} deals={args.deals} pop={args.pop}")

    stale = 0
    t0 = time.time()
    for it in range(int(args.iters)):
        train_seed = int(args.seed) + 123 + it
        val_seed = int(args.seed) + 999 + it

        # generate candidates
        cand_params = [propose(rng, best, sigma) for _ in range(int(args.pop))]
        cand_params.append(best)  # include current best

        scored = []
        for j, pmap in enumerate(cand_params):
            hero = HeroAgent(**pmap)
            wr, _ = paired_winrate(hero, v0, deals=args.deals, seed=train_seed)
            scored.append((wr, j, pmap))

        scored.sort(reverse=True, key=lambda x: x[0])
        top_wr, _, top = scored[0]

        improved = top_wr > best_train + 1e-9
        if improved:
            best = dict(top)
            best_train = float(top_wr)
            best_val, _ = paired_winrate(HeroAgent(**best), v0, deals=args.deals, seed=val_seed)
            stale = 0
        else:
            stale += 1

        # anneal sigma gently
        if (it + 1) % 25 == 0:
            sigma *= 0.85

        if (it % 5) == 0 or improved:
            elapsed = time.time() - t0
            tag = "*" if improved else ""
            print(
                f"it={it:03d} train_seed={train_seed} val_seed={val_seed} best_train={best_train:.4f} best_val={best_val:.4f} sigma={sigma:.3f} stale={stale} {tag} elapsed={elapsed:.1f}s"
            )

        # Early stop conditions
        if best_val >= 0.70 and best_train >= 0.70:
            break
        if stale >= int(args.patience):
            break

    print("---")
    print("BEST PARAMS:")
    for k in sorted(best.keys()):
        print(f"{k}={best[k]:.6f}")


if __name__ == "__main__":
    main()
