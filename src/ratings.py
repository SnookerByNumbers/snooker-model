"""Rating table of every active player: the forecast model's frame-level strength, on an Elo-like scale.

Usage: python -I ratings.py DATA_DIR BACKTEST_JSON AS_OF OUT_JSON [--since YYYY-MM-DD] [--min-matches 20]
                          [--draws 20000] [--seed 0]

Same model and backtest-chosen config as the forecasts (model.py). 1500 is a typical tour pro (the median
rating of players with >= 40 matches in the past year); +100 means winning about 64% of frames against an
equal-rated opponent's 36%, as in Elo. Players need --min-matches matches in the year before AS_OF to be
listed. --since adds each player's rating at that date (e.g. the season start) for a change column.
Also reports raw/time-weighted frame counts and joint-posterior rank intervals/top-K probabilities.
The point-estimate order is retained; uncertainty is not a player-specific rating penalty.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Data, covariance, day, fit  # noqa: E402


def rating_state(d, cfg, as_of, with_sd=True):
    """Fit once and retain the full covariance for correlated rank draws."""
    t_ref = day(as_of)
    mask = (d.t <= t_ref) & (cfg["championship_league"] | ~d.cl)
    x = fit(d, mask, t_ref, cfg["half_life"], cfg["lam"], cfg["home"])
    n = len(d.players)
    recent = mask & (d.t > t_ref - 365)
    counts = np.bincount(d.i[recent], minlength=n) + np.bincount(d.j[recent], minlength=n)
    ref = float(np.median(x[:n][counts >= 40]))
    cov = covariance(d, mask, t_ref, cfg["half_life"], cfg["lam"], cfg["home"], x) if with_sd else None
    sd = np.sqrt(np.diag(cov))[:n] if with_sd else np.zeros(n)
    k = 400 / np.log(10)
    values = {p: (float(1500 + k * (x[i] - ref)), float(k * sd[i]), float(expit(x[i] - ref)), int(counts[i]))
              for i, p in enumerate(d.players)}
    return {"values": values, "ref": ref, "x": x, "cov": cov, "mask": mask, "recent": recent, "t_ref": t_ref}


def table(d, cfg, as_of, with_sd=True):
    """Keep the tuple interface used by the video/statistics tools."""
    state = rating_state(d, cfg, as_of, with_sd)
    return state["values"], state["ref"]


def rank_uncertainty(mean, cov, draws=20000, seed=0):
    """Ranks within the supplied eligible field, using joint Laplace posterior samples.

    The shared covariance matters: independent marginal draws would lose opponent correlations.
    Intervals are model-conditional, not empirically validated confidence bounds.
    """
    if draws < 1000:
        raise ValueError("use at least 1000 posterior draws")
    if len(mean) == 0:
        return []
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean, cov, size=draws)
    ranks = np.argsort(np.argsort(-samples, axis=1), axis=1) + 1
    low, median, high = np.quantile(ranks, [.025, .5, .975], axis=0, method="inverted_cdf")
    return [{"rank_mean": float(ranks[:, i].mean()), "rank_median": int(median[i]),
             "rank_low": int(low[i]), "rank_high": int(high[i]),
             "top5_probability": float((ranks[:, i] <= 5).mean()),
             "top16_probability": float((ranks[:, i] <= 16).mean())} for i in range(len(mean))]


def ranking(d, cfg, as_of, min_matches=20, since=None, draws=20000, seed=0):
    if min_matches < 1:
        raise ValueError("min_matches must be positive")
    state = rating_state(d, cfg, as_of)
    now = state["values"]
    then = table(d, cfg, since, with_sd=False)[0] if since else {}
    rows = sorted(((p, v) for p, v in now.items() if v[3] >= min_matches), key=lambda pv: -pv[1][0])
    slots = [d.idx[p] for p, _ in rows]
    stats = rank_uncertainty(state["x"][slots], state["cov"][np.ix_(slots, slots)], draws, seed)
    n = len(d.players)
    mask, recent = state["mask"], state["recent"]

    def total_frames(selected, weighted=False):
        weights = (d.a + d.b)[selected]
        if weighted:
            weights = weights * .5 ** ((state["t_ref"] - d.t[selected]) / cfg["half_life"])
        return (np.bincount(d.i[selected], weights=weights, minlength=n)
                + np.bincount(d.j[selected], weights=weights, minlength=n))

    frames, weighted, recent_frames = total_frames(mask), total_frames(mask, True), total_frames(recent)
    players = [{"rank": r + 1, "name": p, "nat": d.nat.get(p, ""), "elo": v[0], "elo_sd": v[1],
                "frame_vs_typical_pro": v[2], "matches_past_year": v[3],
                "frames": int(frames[d.idx[p]]), "weighted_frames": float(weighted[d.idx[p]]),
                "frames_past_year": int(recent_frames[d.idx[p]]),
                "elo_since": then[p][0] if p in then and then[p][3] > 0 else None, **stats[r]}
               for r, (p, v) in enumerate(rows)]
    return {"as_of": as_of, "config": cfg, "min_matches": min_matches, "since": since,
            "n_matches_used": int(mask.sum()), "players": players,
            "ranking_method": "point_estimate_with_joint_posterior_uncertainty",
            "ranking_uncertainty": {"draws": draws, "seed": seed, "interval_mass": .95,
                                    "population": "players_meeting_min_matches", "n_players": len(rows),
                                    "method": "joint_laplace_normal",
                                    "note": "Conditional on this model; weighted_frames is time-decayed exposure, not independent sample size."}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("backtest")
    ap.add_argument("as_of")
    ap.add_argument("out")
    ap.add_argument("--since")
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Data(a.data_dir)
    best = min(json.load(open(a.backtest)), key=lambda r: r["log_loss"])
    cfg = {k: best[k] for k in ("half_life", "lam", "home", "championship_league")}
    out = ranking(d, cfg, a.as_of, a.min_matches, a.since, a.draws, a.seed)
    players = out["players"]
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("Point rank; joint posterior 95% rank range is conditional on the model and eligible field.")
    for p in players[:70]:
        ch = f"{p['elo'] - p['elo_since']:+5.0f}" if p["elo_since"] is not None else "   --"
        print(f"{p['rank']:3d} {p['name']:24} {p['nat']:4} {p['elo']:5.0f}±{p['elo_sd']:<3.0f} "
              f"rank95={p['rank_low']:2d}-{p['rank_high']:<2d} top16={p['top16_probability']:6.1%} "
              f"frames={p['frames']:4d} weighted={p['weighted_frames']:6.1f} since {ch}")
    print(f"{len(players)} players with >= {a.min_matches} matches in the year to {a.as_of}")


if __name__ == "__main__":
    main()
