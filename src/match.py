"""Win probability for one match between two players, with the model and config used for the forecasts.

Usage: python -I match.py DATA_DIR BACKTEST_JSON PLAYER_A PLAYER_B --best-of 19 [--as-of YYYY-MM-DD]
                          [--score A B] [--host-country China]

Players are named as in data/matches.csv (English names, e.g. "Judd Trump"). --score gives the frames each player
has already won (a match in progress). --host-country China gives Chinese players the home term, if the backtest
chose to use one (it does not at present). Prints both ratings, the frame-win chance and the match-win chance:
"point" uses the best-estimate ratings, "with uncertainty" averages over 4000 rating draws, as the forecasts do.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Data, forecast  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("backtest")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--best-of", type=int, required=True)
    ap.add_argument("--as-of", help="rate players on matches up to this date (default: the latest match in the data)")
    ap.add_argument("--score", type=int, nargs=2, default=[0, 0], metavar=("FRAMES_A", "FRAMES_B"))
    ap.add_argument("--host-country", default="")
    a = ap.parse_args()
    d = Data(a.data_dir)
    for p in (a.a, a.b):
        if p not in d.idx:
            close = [q for q in d.players if p.split()[-1].lower() in q.lower()][:8]
            sys.exit(f"unknown player {p!r}; names are as in matches.csv" + (f", e.g. {close}" if close else ""))
    as_of = a.as_of or date.fromordinal(int(d.t.max())).isoformat()  # default: the latest match in the data
    best = min(json.load(open(a.backtest)), key=lambda r: r["log_loss"])
    cfg = {k: best[k] for k in ("half_life", "lam", "home", "championship_league")}
    live = [{"p1": a.a, "p2": a.b, "f1": a.score[0], "f2": a.score[1]}] if any(a.score) else []
    draw = {"event": f"{a.a} v {a.b}", "as_of": as_of, "host_country": a.host_country,
            "rounds": [{"name": "Match", "best_of": a.best_of}], "draw": [a.a, a.b], "live": live}
    res = forecast(d, draw, cfg)
    pa, pb = res["players"]
    m = res["first_round"][0]
    print(f"ratings as of {as_of} (config {cfg})")
    for p in (pa, pb):
        print(f"  {p['name']:24} {p['elo']:5.0f} ± {1.96 * p['elo_sd']:.0f} (95%)   "
              f"{p['matches_past_year']} matches in the past year")
    print(f"frame: {a.a} wins a frame {m['frame_p1']:.1%}")
    score = f" from {a.score[0]}-{a.score[1]}" if any(a.score) else ""
    print(f"match, best of {a.best_of}{score}: {a.a} {m['match_p1']:.1%} (with uncertainty), "
          f"{m['match_p1_point']:.1%} (point estimate); {a.b} {1 - m['match_p1']:.1%}")


if __name__ == "__main__":
    main()
