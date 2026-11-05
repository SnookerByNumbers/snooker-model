"""Rating table of every active player: the forecast model's frame-level strength, on an Elo-like scale.

Usage: python -I ratings.py DATA_DIR BACKTEST_JSON AS_OF OUT_JSON [--since YYYY-MM-DD] [--min-matches 20]

Same model and backtest-chosen config as the forecasts (model.py). 1500 is a typical tour pro (the median
rating of players with >= 40 matches in the past year); +100 means winning about 64% of frames against an
equal-rated opponent's 36%, as in Elo. Players need --min-matches matches in the year before AS_OF to be
listed. --since adds each player's rating at that date (e.g. the season start) for a change column.
Writes {"as_of", "config", "reference", "since", "players": [{rank, name, nat, elo, elo_sd,
frame_vs_typical_pro, matches_past_year, elo_since}, ...]} sorted by rating.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Data, covariance, day, fit  # noqa: E402


def table(d, cfg, as_of, with_sd=True):
    """{player: (elo, elo_sd, frame win vs a typical pro, matches in the past year)} as of a date."""
    t_ref = day(as_of)
    mask = (d.t <= t_ref) & (cfg["championship_league"] | ~d.cl)
    x = fit(d, mask, t_ref, cfg["half_life"], cfg["lam"], cfg["home"])
    n = len(d.players)
    recent = mask & (d.t > t_ref - 365)
    counts = np.bincount(d.i[recent], minlength=n) + np.bincount(d.j[recent], minlength=n)
    ref = float(np.median(x[:n][counts >= 40]))
    sd = np.sqrt(np.diag(covariance(d, mask, t_ref, cfg["half_life"], cfg["lam"], cfg["home"], x)))[:n] \
        if with_sd else np.zeros(n)
    k = 400 / np.log(10)
    return {p: (float(1500 + k * (x[i] - ref)), float(k * sd[i]), float(expit(x[i] - ref)), int(counts[i]))
            for i, p in enumerate(d.players)}, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("backtest")
    ap.add_argument("as_of")
    ap.add_argument("out")
    ap.add_argument("--since")
    ap.add_argument("--min-matches", type=int, default=20)
    a = ap.parse_args()
    d = Data(a.data_dir)
    best = min(json.load(open(a.backtest)), key=lambda r: r["log_loss"])
    cfg = {k: best[k] for k in ("half_life", "lam", "home", "championship_league")}
    now, ref = table(d, cfg, a.as_of)
    then = table(d, cfg, a.since, with_sd=False)[0] if a.since else {}
    rows = sorted(((p, v) for p, v in now.items() if v[3] >= a.min_matches), key=lambda pv: -pv[1][0])
    players = [{"rank": r + 1, "name": p, "nat": d.nat.get(p, ""), "elo": v[0], "elo_sd": v[1],
                "frame_vs_typical_pro": v[2], "matches_past_year": v[3],
                "elo_since": then[p][0] if p in then and then[p][3] > 0 else None}
               for r, (p, v) in enumerate(rows)]
    out = {"as_of": a.as_of, "config": cfg, "min_matches": a.min_matches, "since": a.since,
           "n_matches_used": int((d.t <= day(a.as_of)).sum()), "players": players}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    for p in players[:70]:
        ch = f"{p['elo'] - p['elo_since']:+5.0f}" if p["elo_since"] is not None else "   --"
        print(f"{p['rank']:3d} {p['name']:24} {p['nat']:4} {p['elo']:5.0f}±{p['elo_sd']:<3.0f} "
              f"{p['frame_vs_typical_pro']:6.1%} n={p['matches_past_year']:3d} since {ch}")
    print(f"{len(players)} players with >= {a.min_matches} matches in the year to {a.as_of}")


if __name__ == "__main__":
    main()
