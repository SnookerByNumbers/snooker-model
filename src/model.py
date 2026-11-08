"""Frame-level Bradley-Terry ratings with time decay, a walk-forward backtest, and knockout-bracket odds.

Each match contributes its frames: P(i wins a frame vs j) = sigmoid(r_i - r_j + h * (home_i - home_j)),
where home = Chinese player at an event in China. Matches are down-weighted by age (half-life),
and a ridge prior (lam) shrinks players with little data toward the field average.
Match odds follow from frame odds by the race-to-N formula, so one rating serves every format.

Usage:
  python -I model.py backtest DATA_DIR OUT_JSON
  python -I model.py forecast DATA_DIR DRAW_JSON BACKTEST_JSON OUT_JSON
"""
import csv
import json
import sys
from datetime import date

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, log_expit

H_PRIOR = 1e-2  # weak ridge on the home coefficient


def day(s):
    return date.fromisoformat(s).toordinal()


class Data:
    def __init__(self, data_dir):
        rows = list(csv.DictReader(open(f"{data_dir}/matches.csv")))
        self.events = {r["event"]: r for r in csv.DictReader(open(f"{data_dir}/events.csv"))}
        self.players = sorted({r["p1"] for r in rows} | {r["p2"] for r in rows})
        self.idx = {p: k for k, p in enumerate(self.players)}
        self.nat = {}
        for r in rows:
            self.nat[r["p1"]], self.nat[r["p2"]] = r["nat1"], r["nat2"]
        self.i = np.array([self.idx[r["p1"]] for r in rows])
        self.j = np.array([self.idx[r["p2"]] for r in rows])
        self.a = np.array([float(r["f1"]) for r in rows])
        self.b = np.array([float(r["f2"]) for r in rows])
        self.t = np.array([day(r["date"]) for r in rows])
        self.event = np.array([r["event"] for r in rows])
        self.stage = np.array([r.get("stage") or "main" for r in rows])
        china = np.array([r["country"] == "China" for r in rows])
        self.home = (china & np.array([r["nat1"] == "CHN" for r in rows])).astype(float) \
            - (china & np.array([r["nat2"] == "CHN" for r in rows])).astype(float)
        self.cl = np.array(["Championship League" in e for e in self.event])


def fit(d, mask, t_ref, half_life, lam, use_home, x0=None):
    """Return (ratings, home coefficient) fitted on matches in mask, weighted by age at t_ref."""
    n = len(d.players)
    I, J, A, B = d.i[mask], d.j[mask], d.a[mask], d.b[mask]
    W = 0.5 ** (np.maximum(t_ref - d.t[mask], 0) / half_life)
    Hd = d.home[mask] if use_home else np.zeros(mask.sum())

    def f(x):
        r, h = x[:n], x[n]
        z = r[I] - r[J] + h * Hd
        p = expit(z)
        obj = -(W * (A * log_expit(z) + B * log_expit(-z))).sum() + 0.5 * lam * r @ r + 0.5 * H_PRIOR * h * h
        g = W * (A * (1 - p) - B * p)
        grad_r = -(np.bincount(I, g, n) - np.bincount(J, g, n)) + lam * r
        return obj, np.append(grad_r, -(g * Hd).sum() + H_PRIOR * h)

    res = minimize(f, np.zeros(n + 1) if x0 is None else x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-7})
    return res.x


def covariance(d, mask, t_ref, half_life, lam, use_home, x):
    """Laplace approximation: inverse Hessian of the penalised weighted negative log-likelihood."""
    n = len(d.players)
    I, J = d.i[mask], d.j[mask]
    W = 0.5 ** (np.maximum(t_ref - d.t[mask], 0) / half_life)
    Hd = d.home[mask] if use_home else np.zeros(mask.sum())
    z = x[I] - x[J] + x[n] * Hd
    c = W * (d.a[mask] + d.b[mask]) * expit(z) * expit(-z)
    H = np.zeros((n + 1, n + 1))
    np.add.at(H, (I, I), c)
    np.add.at(H, (J, J), c)
    np.add.at(H, (I, J), -c)
    np.add.at(H, (J, I), -c)
    for k, s in ((I, 1.0), (J, -1.0)):
        np.add.at(H, (k, np.full_like(k, n)), s * c * Hd)
        np.add.at(H, (np.full_like(k, n), k), s * c * Hd)
    H[n, n] += (c * Hd * Hd).sum()
    H[np.arange(n), np.arange(n)] += lam
    H[n, n] += H_PRIOR
    cov = np.linalg.inv(H)
    return (cov + cov.T) / 2


def race(p, n, a=0, b=0):
    """P(reaching n frames before the opponent, from a score of a-b, with frame-win probability p)."""
    m1, m2 = n - a, n - b
    if m1 <= 0 or m2 <= 0:
        return np.where(np.asarray(p) >= 0, float(m1 <= 0), 0.0)
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)[..., None]
    k = np.arange(m2)
    logc = gammaln(m1 + k) - gammaln(k + 1) - gammaln(m1)
    return np.exp(logc + m1 * np.log(p) + k * np.log1p(-p)).sum(-1)


def backtest(d, configs, since):
    """Walk forward through events starting on/after `since`: rate players on everything played before
    the event starts, then score the event's main-stage matches (knockout, race-to-N)."""
    tests = sorted((e for e, r in d.events.items()
                    if r["start_date"] >= since and "Championship League" not in e and (d.event == e).any()),
                   key=lambda e: d.events[e]["start_date"])
    out = []
    for hl, lam, use_home, use_cl in configs:
        ys, ps, x = [], [], None
        for e in tests:
            t0 = day(d.events[e]["start_date"])
            train = (d.t < t0) & (use_cl | ~d.cl)
            test = (d.event == e) & (d.stage == "main") & (d.t >= t0) & (d.a != d.b)
            if not test.any():
                continue
            x = fit(d, train, t0, hl, lam, use_home, x)
            n = len(d.players)
            z = x[d.i[test]] - x[d.j[test]] + x[n] * d.home[test]
            nmax = np.maximum(d.a[test], d.b[test]).astype(int)
            ps += [float(race(expit(zz), int(nn))) for zz, nn in zip(z, nmax)]
            ys += list((d.a[test] > d.b[test]).astype(float))
        ys, ps = np.array(ys), np.clip(np.array(ps), 1e-6, 1 - 1e-6)
        rec = {"half_life": hl, "lam": lam, "home": use_home, "championship_league": use_cl,
               "n_matches": len(ys), "n_events": len(tests),
               "log_loss": float(-(ys * np.log(ps) + (1 - ys) * np.log(1 - ps)).mean()),
               "brier": float(((ps - ys) ** 2).mean()),
               "accuracy": float(((ps > 0.5) == (ys > 0.5)).mean())}
        print(json.dumps(rec), flush=True)
        out.append(rec | {"_y": ys.tolist(), "_p": ps.tolist()})
    return out


def bracket_odds(R, rounds, live):
    """Exact knockout odds. R: (..., m, m) for a draw of m = 2**rounds players, frame-win log-odds of row vs column player.
    rounds: race targets per round; live: {(slot_i, slot_j): (frames_i, frames_j)} for first-round matches in play.
    Returns (..., n_rounds, m): probability of winning through each round."""
    P = expit(R)
    size = R.shape[-1]
    alive = np.ones(R.shape[:-1])
    out = []
    for k, n in enumerate(rounds):
        M = race(P, n)
        if k == 0:
            for (si, sj), (fi, fj) in live.items():
                M[..., si, sj] = race(P[..., si, sj], n, fi, fj)
                M[..., sj, si] = 1 - M[..., si, sj]
        block = 2 ** (k + 1)
        new = np.zeros_like(alive)
        for s in range(size):
            lo = (s // block) * block
            half = block // 2
            opp = range(lo + half, lo + block) if (s - lo) < half else range(lo, lo + half)
            opp = list(opp)
            new[..., s] = alive[..., s] * (alive[..., opp] * M[..., s, opp]).sum(-1)
        alive = new
        out.append(alive)
    return np.stack(out, axis=-2)


def forecast(d, draw, cfg, draws=4000, seed=0):
    b = bracket_samples(d, draw, cfg, draws, seed)
    x, cov, n, slots, home, live, point = (b[k] for k in ("x", "cov", "n", "slots", "home", "live", "point"))
    t_ref, full = b["t_ref"], b["samples"].mean(0)
    return finish_forecast(d, draw, cfg, x, cov, n, slots, home, live, point, full, t_ref)


def bracket_samples(d, draw, cfg, draws=4000, seed=0):
    """Fit, then bracket odds at the point estimate and for `draws` rating samples from the Laplace posterior.
    samples has shape (draws, n_rounds, n_slots); forecast() averages it, focus_stats.py uses it for joint odds."""
    hl, lam, use_home = cfg["half_life"], cfg["lam"], cfg["home"]
    t_ref = day(draw["as_of"])
    mask = (d.t <= t_ref) & (cfg["championship_league"] | ~d.cl)
    x = fit(d, mask, t_ref, hl, lam, use_home)
    n = len(d.players)
    cov = covariance(d, mask, t_ref, hl, lam, use_home, x)

    slots = [d.idx[p] for p in draw["draw"]]
    home = np.array([float(draw["host_country"] == "China" and d.nat.get(p) == "CHN") for p in draw["draw"]])
    rounds = [(r["best_of"] + 1) // 2 for r in draw["rounds"]]
    live = {}
    for m in draw.get("live", []):
        si, sj = draw["draw"].index(m["p1"]), draw["draw"].index(m["p2"])
        live[(si, sj)] = (m["f1"], m["f2"])

    def logits(r, h):
        s = r[..., :, None] - r[..., None, :]
        return s + h[..., None, None] * (home[:, None] - home[None, :])

    point = bracket_odds(logits(x[slots], np.array(x[n])), rounds, live)

    rng = np.random.default_rng(seed)
    sub = slots + [n] if use_home else slots  # an unused home term is fixed at 0, not sampled from its prior
    sample = rng.multivariate_normal(x[sub], cov[np.ix_(sub, sub)], size=draws)
    h = sample[:, -1] if use_home else np.zeros(draws)
    samples = bracket_odds(logits(sample[:, :len(slots)], h), rounds, live)
    return {"x": x, "cov": cov, "n": n, "slots": slots, "home": home, "rounds": rounds, "live": live,
            "point": point, "samples": samples, "t_ref": t_ref}


def finish_forecast(d, draw, cfg, x, cov, n, slots, home, live, point, full, t_ref):
    # reference: median rating of players with >= 40 matches in the past year (a typical active tour pro)
    recent = d.t > t_ref - 365
    counts = np.bincount(d.i[recent], minlength=n) + np.bincount(d.j[recent], minlength=n)
    ref = float(np.median(x[:n][counts >= 40]))
    sd = np.sqrt(np.diag(cov))
    elo = lambda r: 1500 + 400 / np.log(10) * (r - ref)

    leaders = sorted(((float(x[k]), d.players[k]) for k in range(n) if counts[k] >= 20), reverse=True)[:25]
    first = []
    for s in range(0, len(slots), 2):
        lo, hi = live.get((s, s + 1), (0, 0))
        z_pt = x[slots[s]] - x[slots[s + 1]] + x[n] * (home[s] - home[s + 1])
        first.append({"p1": draw["draw"][s], "p2": draw["draw"][s + 1], "score": [lo, hi],
                      "frame_p1": float(expit(z_pt)),
                      "match_p1_point": float(point[0, s]),
                      "match_p1": float(full[0, s])})
    return {
        "as_of": draw["as_of"], "event": draw["event"], "config": cfg,
        "home_coef": float(x[n]), "home_coef_sd": float(sd[n]),
        "rounds": [r["name"] for r in draw["rounds"]], "best_of": [r["best_of"] for r in draw["rounds"]],
        "players": [{"name": p, "nat": d.nat.get(p, ""),
                     "elo": float(elo(x[slots[s]])), "elo_sd": float(400 / np.log(10) * sd[slots[s]]),
                     "frame_vs_typical_pro": float(expit(x[slots[s]] - ref)),
                     "matches_past_year": int(counts[slots[s]]),
                     "point": point[:, s].tolist(), "odds": full[:, s].tolist()}
                    for s, p in enumerate(draw["draw"])],
        "first_round": first,
        "leaders": [{"name": p, "elo": float(elo(r))} for r, p in leaders],
    }


def main():
    mode, data_dir = sys.argv[1], sys.argv[2]
    d = Data(data_dir)
    if mode == "backtest":
        since = "2025-07-01"
        grid = [(hl, lam, False, True) for hl in (240, 365, 540, 730, 1095, 1460, 2190) for lam in (2, 4, 8, 16, 32, 64)]
        res = backtest(d, grid + [(1e6, 0.01, False, True)], since)  # last: plain BT, no decay or shrinkage
        best = min(res, key=lambda r: r["log_loss"])
        hl, lam = best["half_life"], best["lam"]
        res += backtest(d, [(hl, lam, True, True), (hl, lam, False, False), (hl, lam, True, False)], since)
        json.dump(res, open(sys.argv[3], "w"))
    else:
        draw = json.load(open(sys.argv[3]))
        bt = [r for r in json.load(open(sys.argv[4]))]
        best = min(bt, key=lambda r: r["log_loss"])
        cfg = {k: best[k] for k in ("half_life", "lam", "home", "championship_league")}
        json.dump(forecast(d, draw, cfg), open(sys.argv[5], "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
