"""
EV / bankroll Monte Carlo simulator for Magnum 4D.

Payouts below reflect approximate Magnum "Big" bet structure per RM1 stake:
  1st  RM 2,500
  2nd  RM 1,000
  3rd  RM   500
  Special (10) RM 180 each
  Consolation (10) RM 60 each

23 prize slots total per draw. Random-pick EV per RM1 bet:
  EV = (2500 + 1000 + 500 + 10*180 + 10*60) / 10000
     = (2500 + 1000 + 500 + 1800 + 600) / 10000
     = 6400 / 10000
     = 0.64
House edge = 1 - 0.64 = 36%.

This is a SIMULATOR not a planner. Use it to visualize loss curves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


BIG_PAYOUT = {
    '1st': 2500,
    '2nd': 1000,
    '3rd': 500,
    'special': 180,
    'consolation': 60,
}
N_SPECIAL = 10
N_CONSOLATION = 10
TOTAL_SLOTS = 1 + 1 + 1 + N_SPECIAL + N_CONSOLATION  # 23


@dataclass
class SimResult:
    pnl_curve: np.ndarray
    bankroll_curve: np.ndarray
    total_bet: float
    total_won: float
    hits_per_tier: dict
    ruined: bool
    n_draws: int

    @property
    def net(self) -> float:
        return float(self.total_won - self.total_bet)

    @property
    def roi(self) -> float:
        return self.net / self.total_bet if self.total_bet > 0 else 0.0


def draw_random_prize_slots(rng: np.random.Generator) -> dict:
    """Sample 23 numbers without replacement for one draw."""
    chosen = rng.choice(10000, size=TOTAL_SLOTS, replace=False)
    return {
        '1st': f'{chosen[0]:04d}',
        '2nd': f'{chosen[1]:04d}',
        '3rd': f'{chosen[2]:04d}',
        'special': [f'{c:04d}' for c in chosen[3:3 + N_SPECIAL]],
        'consolation': [f'{c:04d}' for c in chosen[3 + N_SPECIAL:]],
    }


def evaluate_ticket(number: str, draw: dict) -> int:
    """Return RM payout for one RM1 'Big' bet on `number` against `draw`."""
    if number == draw['1st']:
        return BIG_PAYOUT['1st']
    if number == draw['2nd']:
        return BIG_PAYOUT['2nd']
    if number == draw['3rd']:
        return BIG_PAYOUT['3rd']
    if number in draw['special']:
        return BIG_PAYOUT['special']
    if number in draw['consolation']:
        return BIG_PAYOUT['consolation']
    return 0


def simulate(
    numbers: list[str],
    n_draws: int = 1000,
    bet_per_number: float = 1.0,
    starting_bankroll: float = 1000.0,
    stop_on_ruin: bool = True,
    seed: int = 42,
) -> SimResult:
    """Bet `numbers` (each RM `bet_per_number`) on every draw for n_draws.

    Uses synthetic uniform-random draws (matches the published randomness of
    the lottery; treating real history as ground truth gives the same EV
    asymptotically but smaller samples).
    """
    rng = np.random.default_rng(seed)
    numbers = [str(n).zfill(4) for n in numbers]
    cost_per_draw = bet_per_number * len(numbers)

    pnl = np.zeros(n_draws, dtype=float)
    bank = np.zeros(n_draws, dtype=float)
    hits = {'1st': 0, '2nd': 0, '3rd': 0, 'special': 0, 'consolation': 0}

    bankroll = float(starting_bankroll)
    total_bet = 0.0
    total_won = 0.0
    ruined = False

    for i in range(n_draws):
        if stop_on_ruin and bankroll < cost_per_draw:
            ruined = True
            pnl[i:] = pnl[i - 1] if i > 0 else 0
            bank[i:] = bankroll
            break

        draw = draw_random_prize_slots(rng)
        bankroll -= cost_per_draw
        total_bet += cost_per_draw

        round_winnings = 0.0
        for num in numbers:
            payout = evaluate_ticket(num, draw)
            if payout > 0:
                round_winnings += payout * bet_per_number
                # tier accounting
                if num == draw['1st']:
                    hits['1st'] += 1
                elif num == draw['2nd']:
                    hits['2nd'] += 1
                elif num == draw['3rd']:
                    hits['3rd'] += 1
                elif num in draw['special']:
                    hits['special'] += 1
                elif num in draw['consolation']:
                    hits['consolation'] += 1

        bankroll += round_winnings
        total_won += round_winnings
        pnl[i] = total_won - total_bet
        bank[i] = bankroll

    return SimResult(
        pnl_curve=pnl,
        bankroll_curve=bank,
        total_bet=total_bet,
        total_won=total_won,
        hits_per_tier=hits,
        ruined=ruined,
        n_draws=n_draws,
    )


def theoretical_ev_per_rm1() -> float:
    """Closed-form EV per RM1 stake on one number."""
    payout_sum = (
        BIG_PAYOUT['1st']
        + BIG_PAYOUT['2nd']
        + BIG_PAYOUT['3rd']
        + N_SPECIAL * BIG_PAYOUT['special']
        + N_CONSOLATION * BIG_PAYOUT['consolation']
    )
    return payout_sum / 10000.0


def house_edge() -> float:
    return 1.0 - theoretical_ev_per_rm1()


if __name__ == "__main__":
    print(f"Theoretical EV per RM1: {theoretical_ev_per_rm1():.4f}")
    print(f"House edge: {house_edge() * 100:.2f}%")
    res = simulate(['1234'], n_draws=1000, bet_per_number=1.0, starting_bankroll=10000, seed=7)
    print(f"Net after 1000 draws betting RM1 on 1234: RM {res.net:.2f}")
    print(f"Hits: {res.hits_per_tier}")
    print(f"ROI: {res.roi * 100:.2f}%  Ruined: {res.ruined}")
