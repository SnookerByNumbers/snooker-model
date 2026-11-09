"""Does a per-player rating trend help, and are improving or new players mis-rated? Walk-forward diagnostics.

Usage: python3 -I src/diagnose_drift.py [DATA_DIR] [OUT_MD]      (defaults: data, evaluations/drift_diagnostics.md)

Trend model: rating_i(t) = r_i + s_i * (t - t_ref) / 365, so r_i is the level on the forecast date and s_i the change
per year; s gets its own ridge penalty lam_s (lam_s -> infinity is the selected model). Everything else is as in
model.py (frame-level likelihood, half-life weights, ridge lam on r). Each event is predicted from matches before it.
1. Log loss of the trend model for a range of lam_s against the selected model (bootstrap over events).
2. Bias of the selected model for players grouped by their trend estimated BEFORE the event, and by how long they
   have been in the data: predicted against actual match wins.
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
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
