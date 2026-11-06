"""Describe small-sample support and audit stronger shrinkage on the existing walk-forward results.

Run by evaluate.py. This is an exploratory audit of the existing tuning sample, not a new test set.
Groups use each player's time-weighted frame count BEFORE the event, never the final sample size.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Data, day
from ratings import table


def match_support(d, half_life, use_cl, since="2025-07-01"):
    events = sorted((e for e, r in d.events.items() if r["start_date"] >= since
                     and "Championship League" not in e and (d.event == e).any()),
                    key=lambda e: d.events[e]["start_date"])
    records = []
    n = len(d.players)
    for e in events:
        t = day(d.events[e]["start_date"])
        train = (d.t < t) & (use_cl | ~d.cl)
        test = (d.event == e) & (d.t >= t) & (d.a != d.b)
        weights = (d.a + d.b)[train] * .5 ** ((t - d.t[train]) / half_life)
        counts = (np.bincount(d.i[train], weights=weights, minlength=n)
                  + np.bincount(d.j[train], weights=weights, minlength=n))
        for k in np.flatnonzero(test):
            records.append({"event": e, "p1": d.players[d.i[k]], "p2": d.players[d.j[k]],
                            "y": int(d.a[k] > d.b[k]),
                            "min_weighted_frames": float(min(counts[d.i[k]], counts[d.j[k]]))})
    return records


def audit(root):
    root = Path(root)
    results = json.loads((root / "evaluations/backtest.json").read_text())
    ratings = json.loads((root / "evaluations/ratings.json").read_text())
    best = min(results, key=lambda r: r["log_loss"])
    variants = sorted((r for r in results if all(r[k] == best[k] for k in
                      ("half_life", "home", "championship_league")) and r["lam"] >= best["lam"]),
                      key=lambda r: r["lam"])
    d = Data(root / "data")
    support = match_support(d, best["half_life"], best["championship_league"])
    y = np.array([r["y"] for r in support])
    exposure = np.array([r["min_weighted_frames"] for r in support])
    for r in variants:
        if not np.array_equal(y, r["_y"]) or len(r["_p"]) != len(y):
            raise ValueError("Backtest ordering/schema changed; update support reconstruction before auditing.")
    event_ids = {e: i for i, e in enumerate(sorted({r["event"] for r in support}))}
    ids = np.array([event_ids[r["event"]] for r in support])
    n_events = len(event_ids)
    # Paired resampling of whole events keeps matches within an event together.
    rng = np.random.default_rng(0)
    boots = rng.multinomial(n_events, np.full(n_events, 1 / n_events), size=4000)
    losses = {}
    for r in variants:
        p = np.clip(r["_p"], 1e-6, 1 - 1e-6)
        losses[r["lam"]] = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    baseline = losses[best["lam"]]
    groups = []
    for label, mask in [("全部", np.ones(len(y), bool)), ("不足 250 折算局", exposure < 250),
                        ("250–500 折算局", (exposure >= 250) & (exposure < 500)),
                        ("至少 500 折算局", exposure >= 500)]:
        if not mask.any():
            continue
        counts = np.bincount(ids[mask], minlength=n_events)
        denom = boots @ counts
        comparison = []
        for r in variants:
            loss = losses[r["lam"]]
            sums = np.bincount(ids[mask], weights=(loss - baseline)[mask], minlength=n_events)
            boot_delta = (boots @ sums)[denom > 0] / denom[denom > 0]
            comparison.append({"lam": r["lam"], "log_loss": float(loss[mask].mean()),
                               "delta_vs_selected": float((loss - baseline)[mask].mean()),
                               "event_bootstrap_interval95": np.quantile(boot_delta, [.025, .975]).tolist()})
        groups.append({"group": label, "matches": int(mask.sum()), "variants": comparison})

    examples = [p for p in ratings["players"] if p["name"] in
                ("Chang Bingyu", "Judd Trump", "Zhao Xintong", "Wu Yize")]
    sensitivity = []
    for variant in variants:
        cfg = {k: variant[k] for k in ("half_life", "lam", "home", "championship_league")}
        values = table(d, cfg, ratings["as_of"], with_sd=False)[0]
        eligible = sorted((p for p, v in values.items() if v[3] >= ratings["min_matches"]), key=lambda p: -values[p][0])
        sensitivity.append({"lam": cfg["lam"], "examples": [
            {"name": p["name"], "rank": eligible.index(p["name"]) + 1, "elo": values[p["name"]][0]}
            for p in examples]})
    out = {"as_of": ratings["as_of"], "grouping": "minimum_of_two_players_pre_event_time_weighted_frames",
           "bootstrap": {"unit": "event", "draws": 4000, "seed": 0},
           "selected_config": ratings["config"], "groups": groups, "examples": examples,
           "sensitivity": sensitivity,
           "limitations": ["Same data used for hyperparameter selection; not independent validation.",
                           "Groups and shrinkage comparisons are exploratory, not adjusted for multiple comparisons.",
                           "Weighted frames measure time-decayed exposure, not independent sample size.",
                           "Rank intervals condition on the model and exclude ineligible players."]}
    (root / "evaluations/small_sample.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    names = json.loads((root / "data/names_zh.json").read_text())
    lines = ["# 少样本球员诊断", "", "由 `src/evaluate.py` 调用 `audit_samples.py` 生成。", "",
             f"数据截至 {ratings['as_of']}；样本量、排名和收缩对比都使用当前统一版本。", "",
             "## 局数与排名的不确定性", "",
             "折算局数是按时间衰减后的局数之和，不代表有这么多独立样本。排名区间与前五、前十六概率来自联合后验抽样，",
             f"保留球员间协方差，只在近一年至少打满 {ratings['min_matches']} 场的入榜球员中比较。区间以当前模型假设为条件，未包含模型设定误差。",
             "概率为有限次抽样估计；显示为 0% 或 100% 不表示不可能或必然。", "",
             "| 球员 | 实际局数 | 折算局数 | 点估计排名 | 平均排名 | 约 95% 排名范围 | 前五概率 | 前十六概率 |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    for p in examples:
        lines.append(f"| {names.get(p['name'], p['name'])} | {p['frames']} | {p['weighted_frames']:.1f} | {p['rank']} | "
                     f"{p['rank_mean']:.1f} | {p['rank_low']}–{p['rank_high']} | {p['top5_probability']:.1%} | {p['top16_probability']:.1%} |")
    lines += ["", "## 加大收缩是否改善预测", "",
              "按每站开赛前双方较少的折算局数分组；没有用赛事结束后的样本量分组。对数损失越低越好。",
              "差值为该设置减去当前选中设置，负数才表示改善；区间按赛事整体成对重抽样。", "",
              "| 分组 | 场数 | λ | 对数损失 | 相对当前差值 | 差值约 95% 重抽样区间 |",
              "|---|---:|---:|---:|---:|---|"]
    for group in groups:
        for v in group["variants"]:
            lo, hi = v["event_bootstrap_interval95"]
            lines.append(f"| {group['group']} | {group['matches']} | {v['lam']} | {v['log_loss']:.4f} | "
                         f"{v['delta_vs_selected']:+.4f} | {lo:+.4f} 至 {hi:+.4f} |")
    lines += ["", "## 使用边界", "",
              "当前参数已经做过收缩。不能因为一位球员排名高且局数少，就单独加罚分或只按他一个人的历史表现调参。",
              "应同时展示实力估计、局数和排名区间；预测参数是否修改，要看统一规则在后续数据上的整体与分组表现。",
              "以上比较沿用调参样本，分组和对比属于探索性分析；重抽样区间不是新的独立测试证据，也未校正多重比较。", ""]
    (root / "evaluations/small_sample.md").write_text("\n".join(lines))


if __name__ == "__main__":
    audit(Path(__file__).resolve().parents[1])
