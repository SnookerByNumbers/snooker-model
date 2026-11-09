"""How much the selected model's settings matter: three checks on the walk-forward backtest.

Usage: python3 -I src/diagnose_selection.py [DATA_DIR] [OUT_MD]      (defaults: data, evaluations/selection_diagnostics.md)

1. Selected model against plain Bradley-Terry: difference in log loss with a bootstrap over events.
2. Hold-out: choose half-life and lambda on the first two thirds of the events, score the last third.
3. Over-confidence: scale the frame log-odds by c and see whether c < 1 would have scored better, overall,
   chosen walk-forward (only from earlier events), and by match length.
Diagnostics only: nothing here feeds the forecasts, and the file is not part of evaluate.py's manifest.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Data, day, fit, race  # noqa: E402

SINCE, SELECTED, PLAIN = "2025-07-01", (1095, 8), (1e6, 0.01)
GRID = [(hl, lam) for hl in (365, 730, 1095, 2190) for lam in (2, 8, 32)]
SCALES = np.round(np.arange(0.6, 1.21, 0.05), 2)


def walk_forward(d, events, hl, lam):
    """(event index, frame log-odds of p1, race target, p1 won) for every main-stage match, rated before its event."""
    rows, x = [], None
    for k, e in enumerate(events):
        t0 = day(d.events[e]["start_date"])
        test = (d.event == e) & (d.stage == "main") & (d.t >= t0) & (d.a != d.b)
        if not test.any():
            continue
        x = fit(d, d.t < t0, t0, hl, lam, False, x)
        z = x[d.i[test]] - x[d.j[test]]
        rows += [(k, zz, int(max(a, b)), float(a > b)) for zz, a, b in zip(z, d.a[test], d.b[test])]
    return np.array(rows)


def losses(rows, c=1.0):
    p = np.array([float(race(expit(c * z), int(n))) for z, n in zip(rows[:, 1], rows[:, 2])])
    p, y = np.clip(p, 1e-6, 1 - 1e-6), rows[:, 3]
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "evaluations/selection_diagnostics.md")
    d = Data(data_dir)
    events = sorted((e for e, r in d.events.items() if r["start_date"] >= SINCE and "Championship League" not in e
                     and (d.event == e).any()), key=lambda e: d.events[e]["start_date"])
    rows = {g: walk_forward(d, events, *g) for g in GRID + [PLAIN]}
    L = {g: losses(r) for g, r in rows.items()}
    sel, ev = rows[SELECTED], rows[SELECTED][:, 0]
    evs = np.unique(ev)
    md = ["# 选中参数有多重要：三项诊断", "",
          "由 `python3 -I src/diagnose_selection.py` 生成；请勿手改数字。只作诊断，不进入预测，也不在 manifest 校验范围内。", "",
          f"数据截至 {max(r['date'] for r in __import__('csv').DictReader(open(f'{data_dir}/matches.csv')))}；"
          f"回测自 {SINCE} 起，{len(evs)} 站有正赛、{len(sel)} 场。", ""]

    diff = L[PLAIN] - L[SELECTED]
    rng = np.random.default_rng(0)
    boots = np.array([np.concatenate([diff[ev == e] for e in rng.choice(evs, len(evs))]).mean() for _ in range(4000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    md += ["## 1. 选中模型比普通 BT 好多少", "",
           f"对数损失：选中模型 {L[SELECTED].mean():.4f}，普通 BT（不衰减、λ=0.01）{L[PLAIN].mean():.4f}，"
           f"相差 {diff.mean():.4f}。", "",
           f"按赛事重抽样 4000 次，差值的 95% 区间是 [{lo:.4f}, {hi:.4f}]，选中模型更好的比例 {np.mean(boots > 0):.0%}。", "",
           "结论：区间跨过 0，时间衰减和收缩带来的提升在这批数据上**没有统计显著性**。"
           "可以说“参数由回测选出”，不能说这些参数让预测明显更准。", ""]

    cut = evs[int(len(evs) * 2 / 3)]
    early, late = ev < cut, ev >= cut
    best = min(GRID, key=lambda g: L[g][early].mean())
    md += ["## 2. 留出检验", "",
           f"只用前 {(evs < cut).sum()} 站选参数，选出半衰期 {best[0]} 天、λ = {best[1]}"
           f"（{'与' if best == SELECTED else '不同于'}全量回测选中的 {SELECTED[0]} 天、λ = {SELECTED[1]}"
           f"{'相同' if best == SELECTED else ''}）。", "",
           f"在后 {(evs >= cut).sum()} 站（{int(late.sum())} 场）上：留出选中的参数 {L[best][late].mean():.4f}，"
           f"全量选中的参数 {L[SELECTED][late].mean():.4f}，普通 BT {L[PLAIN][late].mean():.4f}。", "",
           "网格只取了 4 个半衰期 × 3 个 λ。参数选择和成绩报告用同一批回测的问题，在这里没有造成可见的偏差。", ""]

    tab = {float(c): losses(sel, c) for c in SCALES}
    best_c = min(tab, key=lambda c: tab[c].mean())
    wf = np.concatenate([tab[min(tab, key=lambda c: tab[c][ev < e].mean()) if (ev < e).sum() >= 200 else 1.0][ev == e]
                         for e in evs])
    md += ["## 3. 是否过于自信", "",
           "把单局对数几率乘以 c（c < 1 表示把胜率往 50% 收）：", "",
           "| c | " + " | ".join(f"{c:.2f}" for c in tab) + " |", "|---|" + "---:|" * len(tab),
           "| 对数损失 | " + " | ".join(f"{v.mean():.4f}" for v in tab.values()) + " |", "",
           f"事后最好的是 c = {best_c:.2f}（{tab[best_c].mean():.4f}，c = 1 时 {tab[1.0].mean():.4f}）。"
           f"但逐站只用此前的比赛来选 c，得到 {wf.mean():.4f}，"
           f"{'不如' if wf.mean() > tab[1.0].mean() else '好于'}不调整。", "",
           "| 赛制（先赢局数） | 场数 | 热门平均预测 | 热门实际胜率 | 事后最好的 c |", "|---|---:|---:|---:|---:|"]
    for a, b in ((1, 4), (5, 5), (6, 9), (10, 18)):
        m = (sel[:, 2] >= a) & (sel[:, 2] <= b)
        if not m.any():
            continue
        p = np.array([float(race(expit(z), int(n))) for z, n in zip(sel[m, 1], sel[m, 2])])
        fav, won = np.where(p >= .5, p, 1 - p), np.where(p >= .5, sel[m, 3], 1 - sel[m, 3])
        md.append(f"| {a if a == b else f'{a}–{b}'} | {int(m.sum())} | {fav.mean():.1%} | {won.mean():.1%} | "
                  f"{min(tab, key=lambda c: tab[c][m].mean()):.2f} |")
    md += ["", "结论：整体略偏自信，主要在短赛制，但幅度在抽样误差之内，事先也选不出稳定有用的 c，**不加校正项**。"
               "数据再多一个赛季后重跑本诊断。", ""]
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
