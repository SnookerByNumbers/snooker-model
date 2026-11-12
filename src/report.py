"""Run the forecast for a draw with the best backtested config and print a summary with sanity checks.

Usage: python -I report.py DATA_DIR DRAW_JSON BACKTEST_JSON OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Data, bracket_odds, covariance, day, fit, forecast, selected  # noqa: E402


def calibration(ys, ps, edges=(0, .2, .35, .5, .65, .8, 1.0001)):
    ys, ps = np.array(ys), np.array(ps)
    # fold to the favourite's perspective so each bin pools both sides of a match
    fav = np.where(ps >= .5, ps, 1 - ps)
    won = np.where(ps >= .5, ys, 1 - ys)
    rows = []
    for lo, hi in zip((.5, .6, .7, .8, .9), (.6, .7, .8, .9, 1.0001)):
        m = (fav >= lo) & (fav < hi)
        if m.any():
            rows.append((lo, min(hi, 1), int(m.sum()), float(fav[m].mean()), float(won[m].mean())))
    return rows


def title_odds_point(d, draw, cfg):
    t_ref = day(draw["as_of"])
    mask = (d.t <= t_ref) & (cfg["championship_league"] | ~d.cl)
    x = fit(d, mask, t_ref, cfg["half_life"], cfg["lam"], cfg["home"])
    n = len(d.players)
    slots = [d.idx[p] for p in draw["draw"]]
    home = np.array([float(draw["host_country"] == "China" and d.nat.get(p) == "CHN") for p in draw["draw"]])
    R = x[slots][:, None] - x[slots][None, :] + x[n] * (home[:, None] - home[None, :])
    rounds = [(r["best_of"] + 1) // 2 for r in draw["rounds"]]
    return bracket_odds(R, rounds, {})[-1], x, mask


def main():
    data_dir, draw_path, bt_path, out_path = sys.argv[1:5]
    d = Data(data_dir)
    draw = json.load(open(draw_path))
    bt = json.load(open(bt_path))
    best = selected(bt)
    cfg = {k: best[k] for k in ("half_life", "lam", "home", "championship_league")}

    res = forecast(d, draw, cfg)
    res["backtest"] = {k: best[k] for k in ("n_events", "n_matches", "log_loss", "brier", "accuracy")}
    json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)

    print(f"config: {cfg}")
    print(f"backtest ({best['n_events']} events, {best['n_matches']} main-stage matches since 2025-07): "
          f"log loss {best['log_loss']:.4f} (coin flip 0.6931), Brier {best['brier']:.4f}, accuracy {best['accuracy']:.3f}")
    print("\ncalibration (favourite's predicted vs actual win rate):")
    for lo, hi, n, pred, act in calibration(best["_y"], best["_p"]):
        print(f"  {lo:.0%}-{hi:.0%}: n={n:4d} predicted {pred:.1%} actual {act:.1%}")

    print(f"\n{draw['rounds'][0]['name']} matches:")
    for m in res["first_round"]:
        print(f"  {m['p1']:>16} {m['match_p1']:6.1%} - {1 - m['match_p1']:6.1%} {m['p2']:<16} "
              f"(frame {m['frame_p1']:.1%})")

    print(f"\n{'player':16} {'elo':>10} {'frame%':>7} " + " ".join(f"{r[:6]:>7}" for r in [r["name"] for r in draw["rounds"][1:]] + ["Win"]))
    for p in sorted(res["players"], key=lambda p: -p["odds"][-1]):
        print(f"{p['name']:16} {p['elo']:5.0f}±{p['elo_sd']:<3.0f} {p['frame_vs_typical_pro']:7.1%} "
              + " ".join(f"{v:7.1%}" for v in p["odds"]))

    print("\nsensitivity of title odds to the model config (point estimates):")
    alts = {"best": cfg,
            "1y half-life": cfg | {"half_life": 365, "lam": 4},
            "plain BT": cfg | {"half_life": 1e6, "lam": 0.01},
            "with home": cfg | {"home": True}}
    table = {k: title_odds_point(d, draw, c)[0] for k, c in alts.items()}
    print(f"{'player':16} " + " ".join(f"{k:>13}" for k in table))
    order = np.argsort(-table["best"])
    for s in order:
        print(f"{draw['draw'][s]:16} " + " ".join(f"{v[s]:13.1%}" for v in table.values()))

    _, x, mask = title_odds_point(d, draw, alts["with home"])
    t_ref = day(draw["as_of"])
    cov = covariance(d, mask, t_ref, cfg["half_life"], cfg["lam"], True, x)
    n = len(d.players)
    h, sd = x[n], np.sqrt(cov[n, n])
    print(f"\nhome coefficient (Chinese player at an event in China), frame log-odds: {h:+.3f} ± {sd:.3f}"
          f"  -> frame win% vs equal opponent {1 / (1 + np.exp(-h)):.1%}")

    print("\ntop 25 by rating (>= 20 matches in the past year):")
    print("  " + ", ".join(f"{r['name']} {r['elo']:.0f}" for r in res["leaders"]))


if __name__ == "__main__":
    main()
