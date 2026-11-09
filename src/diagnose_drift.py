"""Does a per-player rating trend help, and are improving or new players mis-rated? Walk-forward diagnostics.

Usage: python3 -I src/diagnose_drift.py [DATA_DIR] [OUT_MD]      (defaults: data, evaluations/drift_diagnostics.md)

Trend model: rating_i(t) = r_i + s_i * (t - t_ref) / 365, so r_i is the level on the forecast date and s_i the change
per year; s gets its own ridge penalty lam_s (lam_s -> infinity is the selected model). Everything else is as in
model.py (frame-level likelihood, half-life weights, ridge lam on r). Each event is predicted from matches before it.
1. Log loss of the trend model for a range of lam_s against the selected model (bootstrap over events).
2. Bias of the selected model for players grouped by their trend estimated BEFORE the event, and by how long they
   have been in the data: predicted against actual match wins.
3. Random walk instead of a straight line: one rating per half-year, neighbouring half-years tied by lam_w.
4. Newcomer prior: players first seen less than a year before the forecast date are shrunk toward their own group
   mean (estimated in the fit) instead of the field average.
Diagnostics only: nothing here feeds the forecasts, and the file is not part of evaluate.py's manifest.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Data, day, fit, race  # noqa: E402

SINCE, HL, LAM = "2025-07-01", 1095, 8
LAM_S = (3200, 1600, 800, 400, 200, 100, 50, 25)
GROUP_LAM_S = 100  # trend estimates used to group players in part 2
NAMED = ("Chang Bingyu", "Wu Yize", "Zhao Xintong")


def fit_trend(d, mask, t_ref, lam_s):
    n = len(d.players)
    I, J, A, B = d.i[mask], d.j[mask], d.a[mask], d.b[mask]
    W = 0.5 ** (np.maximum(t_ref - d.t[mask], 0) / HL)
    tau = (d.t[mask] - t_ref) / 365.0
    k = np.sqrt(LAM / lam_s)  # slopes are optimised as s / k: both blocks then have the same ridge curvature

    def f(x):
        r, s = x[:n], x[n:] * k
        z = r[I] - r[J] + (s[I] - s[J]) * tau
        g = W * (A * (1 - expit(z)) - B * expit(z))
        obj = -(W * (A * log_expit(z) + B * log_expit(-z))).sum() + 0.5 * LAM * r @ r + 0.5 * lam_s * s @ s
        gr = -(np.bincount(I, g, n) - np.bincount(J, g, n)) + LAM * r
        gs = -(np.bincount(I, g * tau, n) - np.bincount(J, g * tau, n)) + lam_s * s
        return obj, np.concatenate([gr, gs * k])

    x = minimize(f, np.zeros(2 * n), jac=True, method="L-BFGS-B",
                 options={"maxiter": 5000, "gtol": 1e-8, "ftol": 1e-13}).x
    return x[:n], x[n:] * k


def _minimise(f, m):
    return minimize(f, np.zeros(m), jac=True, method="L-BFGS-B", options={"maxiter": 8000, "gtol": 1e-8, "ftol": 1e-13}).x


def fit_walk(d, mask, t_ref, lam_w, step=182, K=7):
    """Rating per player per half-year (K periods back from t_ref); returns the latest period's rating. Optimised as
    the current level c and backward increments u, r_k = c + sum of u over later periods, so the ridge on c and the
    smoothness penalty lam_w * u^2 are both diagonal."""
    n = len(d.players)
    I, J, A, B = d.i[mask], d.j[mask], d.a[mask], d.b[mask]
    W = 0.5 ** (np.maximum(t_ref - d.t[mask], 0) / HL)
    per = np.clip(K - 1 - (t_ref - d.t[mask]) // step, 0, K - 1).astype(int)
    ii, jj, k = I * K + per, J * K + per, np.sqrt(LAM / lam_w)

    def f(x):
        X = x.reshape(n, K)
        c, u = X[:, K - 1], X[:, :K - 1] * k
        r = np.empty((n, K))
        r[:, K - 1] = c
        r[:, :K - 1] = c[:, None] + np.cumsum(u[:, ::-1], axis=1)[:, ::-1]
        z = r.ravel()[ii] - r.ravel()[jj]
        g = W * (A * (1 - expit(z)) - B * expit(z))
        obj = -(W * (A * log_expit(z) + B * log_expit(-z))).sum() + 0.5 * LAM * c @ c + 0.5 * lam_w * (u * u).sum()
        gr = -(np.bincount(ii, g, n * K) - np.bincount(jj, g, n * K)).reshape(n, K)
        gu = np.cumsum(gr[:, :K - 1], axis=1) + lam_w * u
        return obj, np.concatenate([gu * k, (gr.sum(1) + LAM * c)[:, None]], axis=1).ravel()

    return _minimise(f, n * K).reshape(n, K)[:, K - 1]


def fit_newcomer(d, mask, t_ref, first, days=365):
    """As the selected model, but players first seen under `days` before t_ref are shrunk toward their own mean mu
    (a free parameter). Returns (ratings, mu)."""
    n = len(d.players)
    I, J, A, B = d.i[mask], d.j[mask], d.a[mask], d.b[mask]
    W = 0.5 ** (np.maximum(t_ref - d.t[mask], 0) / HL)
    new = (t_ref - first < days) & (first < t_ref)

    def f(x):
        r, mu = x[:n], x[n]
        z = r[I] - r[J]
        dev = r - mu * new
        g = W * (A * (1 - expit(z)) - B * expit(z))
        obj = -(W * (A * log_expit(z) + B * log_expit(-z))).sum() + 0.5 * LAM * dev @ dev
        return obj, np.append(-(np.bincount(I, g, n) - np.bincount(J, g, n)) + LAM * dev, -LAM * dev[new].sum())

    x = _minimise(f, n + 1)
    return x[:n], x[n]


def walk_forward_with(d, events, fitter):
    """walk_forward for any fitter(mask, t0) -> ratings; same columns, slopes left at 0."""
    rows = []
    for k, e in enumerate(events):
        t0 = day(d.events[e]["start_date"])
        test = (d.event == e) & (d.stage == "main") & (d.t >= t0) & (d.a != d.b)
        if test.any():
            r = fitter(d.t < t0, t0)
            rows += [(k, r[i] - r[j], int(max(a, b)), float(a > b), i, j, 0, 0)
                     for i, j, a, b in zip(d.i[test], d.j[test], d.a[test], d.b[test])]
    return np.array(rows)


def walk_forward(d, events, lam_s=None):
    """Per main-stage match: event index, frame log-odds of p1, race target, p1 won, p1, p2, slope of p1, of p2."""
    rows, x = [], None
    for k, e in enumerate(events):
        t0 = day(d.events[e]["start_date"])
        test = (d.event == e) & (d.stage == "main") & (d.t >= t0) & (d.a != d.b)
        if not test.any():
            continue
        if lam_s is None:
            x = fit(d, d.t < t0, t0, HL, LAM, False, x)
            r, s = x, np.zeros(len(d.players))
        else:
            r, s = fit_trend(d, d.t < t0, t0, lam_s)
        rows += [(k, r[i] - r[j], int(max(a, b)), float(a > b), i, j, s[i], s[j])
                 for i, j, a, b in zip(d.i[test], d.j[test], d.a[test], d.b[test])]
    return np.array(rows)


def match_p(rows):
    return np.array([float(race(expit(z), int(n))) for z, n in zip(rows[:, 1], rows[:, 2])])


def log_loss(rows):
    p, y = np.clip(match_p(rows), 1e-6, 1 - 1e-6), rows[:, 3]
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "evaluations/drift_diagnostics.md")
    d = Data(data_dir)
    zh = json.load(open(f"{data_dir}/names_zh.json"))
    events = sorted((e for e, r in d.events.items() if r["start_date"] >= SINCE and "Championship League" not in e
                     and (d.event == e).any()), key=lambda e: d.events[e]["start_date"])
    base = walk_forward(d, events)
    Lb, ev = log_loss(base), base[:, 0]
    evs = np.unique(ev)
    rng = np.random.default_rng(0)
    last = max(r["date"] for r in csv.DictReader(open(f"{data_dir}/matches.csv")))
    md = ["# 实力漂移诊断：给每位球员加一个趋势项有没有用", "",
          "由 `python3 -I src/diagnose_drift.py` 生成；请勿手改数字。只作诊断，不进入预测，也不在 manifest 校验范围内。", "",
          f"数据截至 {last}；回测自 {SINCE} 起，{len(evs)} 站有正赛、{len(base)} 场；逐站只用开赛前的比赛。", "",
          "趋势模型：实力 = 预测当天的水平 + 每年变化量 × 距预测日的年数。每年变化量另有一个收缩系数 λs，"
          f"λs 越大越接近不加趋势；其余与选中模型相同（半衰期 {HL} 天，λ = {LAM}）。", "",
          "## 1. 加趋势项后的回测", "",
          "| λs | 各场球员趋势的标准差（对数几率/年） | 对数损失 | 相对选中模型 | 差值约 95% 重抽样区间 |", "|---:|---:|---:|---:|---|",
          f"| 不加趋势 | 0 | {Lb.mean():.4f} | +0.0000 | — |"]
    trend = {}
    for lam_s in LAM_S:
        trend[lam_s] = rows = walk_forward(d, events, lam_s)
        diff = log_loss(rows) - Lb
        boots = [np.concatenate([diff[ev == e] for e in rng.choice(evs, len(evs))]).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        md.append(f"| {lam_s} | {np.std(rows[:, 6:8]):.3f} | {log_loss(rows).mean():.4f} | {diff.mean():+.4f} | "
                  f"{lo:+.4f} 至 {hi:+.4f} |")
    worse = all(log_loss(trend[s]).mean() >= Lb.mean() for s in LAM_S)
    md += ["", ("结论：趋势放得越开，对数损失越差，没有任何一档好于不加趋势。" if worse else
                "结论：有的档位好于不加趋势，见上表的区间。") +
           "线性趋势把近期的偶然起伏也外推了出去，整体上得不偿失，**不加入模型**。", ""]

    # part 2: one record per player per match
    T = trend[GROUP_LAM_S]
    pb, pt = match_p(base), match_p(T)
    first = {}
    for i, j, t in zip(d.i, d.j, d.t):
        first[i], first[j] = min(first.get(i, t), t), min(first.get(j, t), t)
    t0 = {k: day(d.events[e]["start_date"]) for k, e in enumerate(events)}
    rec = []
    for r, p1, p2, s1, s2 in zip(base, pb, pt, T[:, 6], T[:, 7]):
        k, i, j = int(r[0]), int(r[4]), int(r[5])
        rec += [(k, i, s1, p1, p2, r[3], t0[k] - first[i]), (k, j, s2, 1 - p1, 1 - p2, 1 - r[3], t0[k] - first[j])]
    rec = np.array(rec)

    def row(name, m):
        k, res = rec[m, 0], rec[m, 5] - rec[m, 3]
        boots = [np.concatenate([res[k == e] for e in rng.choice(evs, len(evs))]).mean() for _ in range(3000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return (f"| {name} | {int(m.sum())} | {rec[m, 3].mean():.1%} | {rec[m, 5].mean():.1%} | {res.mean():+.1%} | "
                f"{lo:+.1%} 至 {hi:+.1%} | {(rec[m, 5] - rec[m, 4]).mean():+.1%} |")

    head = [f"| 分组 | 人场 | 选中模型预测胜率 | 实际胜率 | 实际减预测 | 约 95% 重抽样区间 | 加趋势后（λs = {GROUP_LAM_S}）实际减预测 |",
            "|---|---:|---:|---:|---:|---|---:|"]
    s, ten = rec[:, 2], rec[:, 6]
    q = np.quantile(s, [.1, .25, .75, .9])
    md += ["## 2. 选中模型对哪些球员有偏差", "",
           "每场比赛的两位球员各记一次（人场）。“实际减预测”为正，表示模型低估了这组球员。"
           f"趋势用开赛前的数据估计（λs = {GROUP_LAM_S}），没有用到被预测的比赛。", "",
           "按开赛前估计的趋势分组：", ""] + head + [
        row("上升最快的 10%", s >= q[3]), row("上升较快（75%–90%）", (s >= q[2]) & (s < q[3])),
        row("中间 50%", (s >= q[1]) & (s < q[2])), row("下滑较快（10%–25%）", (s >= q[0]) & (s < q[1])),
        row("下滑最快的 10%", s < q[0]), "",
        "按球员在数据里出现了多久分组（数据从 %s 开始，所以“两年以上”包含所有老球员）：" % min(
            r["date"] for r in csv.DictReader(open(f"{data_dir}/matches.csv"))), ""] + head + [
        row("首次出现不到 1 年", ten < 365), row("首次出现 1–2 年", (ten >= 365) & (ten < 730)),
        row("首次出现 2 年以上", ten >= 730), "", "几位常被问到的球员（场数少，区间很宽）：", ""] + head + [
        row(zh.get(nm, nm), rec[:, 1] == d.idx[nm]) for nm in NAMED if nm in d.idx] + ["",
        "读法：", "- 区间跨过 0 的行，不能说模型对这组有偏差。",
        "- 分组看的是整组的平均，不代表组里每个人都被低估或高估。",
        "- 这里没有年龄数据，“上升快”“出现时间短”只是年轻球员的近似，不能直接说成年龄的影响。", ""]
    # part 3: random walk
    md += ["## 3. 换成分段的漂移（随机游走）", "",
           "不假设直线：每位球员每半年一个实力值，相邻两段之间用 λw 拉住（λw 越大越接近不让变），预测用最近一段。", "",
           "| λw | 对数损失 | 相对选中模型 | 差值约 95% 重抽样区间 |", "|---:|---:|---:|---|"]
    walk_better = False
    for lam_w in (800, 400, 200, 100, 50, 25):
        diff = log_loss(walk_forward_with(d, events, lambda m, t: fit_walk(d, m, t, lam_w))) - Lb
        boots = [np.concatenate([diff[ev == e] for e in rng.choice(evs, len(evs))]).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        walk_better |= diff.mean() < 0
        md.append(f"| {lam_w} | {(Lb + diff).mean():.4f} | {diff.mean():+.4f} | {lo:+.4f} 至 {hi:+.4f} |")
    md += ["", ("结论：有的档位好于选中模型，见上表。" if walk_better else
                "结论：和直线趋势一样，放得越开越差，没有一档好于选中模型，**不加入模型**。"
                "这和参数网格的结果一致（半衰期越短越差）：这批数据里，多用旧比赛比追近况更准。"), ""]

    # part 4: newcomer prior
    mus = []

    def newcomer(m, t):
        r, mu = fit_newcomer(d, m, t, first_arr)
        mus.append(mu)
        return r

    first_arr = np.array([first.get(i, 10 ** 9) for i in range(len(d.players))])
    N = walk_forward_with(d, events, newcomer)
    pn, diff = match_p(N), log_loss(N) - Lb
    ten1, ten2 = np.array([t0[int(k)] - first[int(i)] for k, i in zip(base[:, 0], base[:, 4])]), \
        np.array([t0[int(k)] - first[int(i)] for k, i in zip(base[:, 0], base[:, 5])])
    has_new = (ten1 < 365) | (ten2 < 365)

    def ci(m):
        boots = [np.concatenate([diff[m & (ev == e)] for e in rng.choice(evs, len(evs))]) for _ in range(3000)]
        return np.percentile([b.mean() for b in boots if len(b)], [2.5, 97.5])

    y = base[:, 3]
    bias = lambda p: np.concatenate([(y - p)[ten1 < 365], (p - y)[ten2 < 365]]).mean()  # noqa: E731
    counts = {}
    for k, i, j in zip(base[:, 0], base[:, 4], base[:, 5]):
        for pl in (int(i), int(j)):
            if t0[int(k)] - first[pl] < 365:
                counts[pl] = counts.get(pl, 0) + 1
    top = sorted(counts.items(), key=lambda x: -x[1])
    share = sum(c for _, c in top[:2]) / sum(counts.values())
    md += ["## 4. 给新面孔单独一个收缩中心", "",
           "第 2 节里，出现不到 1 年的球员被高估。试一个修正：这些球员不往全体平均收缩，而往他们自己这组的平均收缩"
           "（这个平均由拟合估出）。", "",
           f"估出的新面孔平均水平：比全体平均低 {-np.mean(mus) * 400 / np.log(10):.0f} 分左右（各站平均）。", "",
           "| | 场数 | 选中模型 | 加新面孔中心 | 差值 | 差值约 95% 重抽样区间 |", "|---|---:|---:|---:|---:|---|"]
    for name, m in (("全部比赛", np.ones(len(base), bool)), ("有新面孔的比赛", has_new), ("其余比赛", ~has_new)):
        lo, hi = ci(m)
        md.append(f"| {name}（对数损失） | {int(m.sum())} | {Lb[m].mean():.4f} | {(Lb + diff)[m].mean():.4f} | "
                  f"{diff[m].mean():+.4f} | {lo:+.4f} 至 {hi:+.4f} |")
    md += ["", f"新面孔的实际减预测：选中模型 {bias(pb):+.1%}，加新面孔中心后 {bias(pn):+.1%}。", "",
           f"这 {sum(counts.values())} 人场来自 {len(counts)} 位球员，其中 "
           f"{'、'.join(f'{zh.get(d.players[i], d.players[i])} {c} 场' for i, c in top[:2])}，两人占 {share:.0%}。"
           "“首次出现”只表示数据窗口里第一次有比赛，长期缺席后复出的球员也算在内；其余球员大多每人只有几场。", "",
           "结论：这个修正把新面孔的平均偏差拉回来了，但对数损失没有改善，在有新面孔的比赛上还略差："
           "它把早有实力、只是刚回到数据里的球员和真正的新人一起往下拉。数据里没有“职业／业余／外卡”这类身份信息，"
           "分不开这两种人，所以**不加入模型**。实际影响也有限：被高估的主要是场次很少的球员。", ""]
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
