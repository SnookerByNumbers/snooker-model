"""Rebuild the tracked evaluation, or verify that it matches the current data and code.

Run from any directory: python -I src/evaluate.py [--check]
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ["data/matches.csv", "data/events.csv", "data/centuries.csv", "data/names_zh.json"]
INPUTS += ["src/" + name for name in (
    "model.py", "parse_matches.py", "parse_centuries.py", "fetch_wiki.py",
    "report.py", "ratings.py", "match.py", "audit_samples.py", "evaluate.py")]
OUTPUTS = ["evaluations/backtest.json", "evaluations/summary.md", "evaluations/ratings.json",
           "evaluations/small_sample.json", "evaluations/small_sample.md"]


def hashes(paths):
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if inputs or outputs changed since evaluation")
    args = ap.parse_args()
    manifest_path = ROOT / "evaluations/manifest.json"
    if args.check:
        saved = json.loads(manifest_path.read_text())
        actual = hashes(INPUTS + OUTPUTS)
        changed = [p for p, digest in actual.items() if saved["sha256"].get(p) != digest]
        if changed:
            raise SystemExit("Evaluation is stale; run src/evaluate.py. Changed: " + ", ".join(changed))
        print("Evaluation matches all tracked inputs and outputs.")
        return

    (ROOT / "evaluations").mkdir(exist_ok=True)
    before = hashes(INPUTS)
    subprocess.run([sys.executable, "-I", str(ROOT / "src/model.py"), "backtest",
                    str(ROOT / "data"), str(ROOT / OUTPUTS[0])], check=True,
                   env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
    if hashes(INPUTS) != before:
        raise SystemExit("Inputs changed while evaluating; rerun before committing.")
    records = json.loads((ROOT / OUTPUTS[0]).read_text())
    best = min(records, key=lambda r: r["log_loss"])
    plain = next(r for r in records if r["half_life"] == 1e6)
    with (ROOT / "data/matches.csv").open() as f:
        matches = list(csv.DictReader(f))
    stats = {
        "matches": len(matches),
        "frames": sum(int(r["f1"]) + int(r["f2"]) for r in matches),
        "players": len({r[k] for r in matches for k in ("p1", "p2")}),
        "first_date": min(r["date"] for r in matches),
        "last_date": max(r["date"] for r in matches),
    }
    sys.path.insert(0, str(ROOT / "src"))
    from report import calibration
    from ratings import ranking
    from model import Data
    from audit_samples import audit
    cfg = {k: best[k] for k in ("half_life", "lam", "home", "championship_league")}
    current_ratings = ranking(Data(ROOT / "data"), cfg, stats["last_date"])
    (ROOT / "evaluations/ratings.json").write_text(json.dumps(current_ratings, ensure_ascii=False, indent=2) + "\n")
    audit(ROOT)
    lines = ["# 当前数据与回测", "", "由 `python3 -I src/evaluate.py` 生成；请勿手改数字。",
             "输入与结果的 SHA-256 校验值见 [manifest.json](manifest.json)。", "",
             f"数据：{stats['matches']} 场、{stats['frames']} 局、{stats['players']} 名球员；"
             f"日期 {stats['first_date']} 至 {stats['last_date']}。", "",
             f"回测：自 2025-07-01 起，{best['n_events']} 站、{best['n_matches']} 场；"
             "逐站使用开赛前的数据，预测该站正赛。", "",
             f"选中参数：半衰期 {best['half_life']} 天，λ = {best['lam']}，"
             f"主场项 {'启用' if best['home'] else '不启用'}，"
             f"冠军联赛训练数据 {'保留' if best['championship_league'] else '排除'}。", "",
             "| 模型 | 对数损失 ↓ | Brier ↓ | 命中率 ↑ |", "|---|---:|---:|---:|",
             "| 全部预测 50% | 0.6931 | 0.2500 | — |"]
    for title, r in [("普通 BT（不衰减、λ=0.01）", plain), ("选中模型", best)]:
        lines.append(f"| {title} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['accuracy']:.1%} |")
    lines += ["", "## 校准", "", "| 热门预测胜率 | 场数 | 平均预测 | 实际胜率 |", "|---|---:|---:|---:|"]
    for lo, hi, n, p, y in calibration(best["_y"], best["_p"]):
        lines.append(f"| {lo:.0%}–{hi:.0%} | {n} | {p:.1%} | {y:.1%} |")
    lines += ["", "## 参数网格", "", "| 半衰期 / λ | 2 | 4 | 8 | 16 | 32 | 64 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for hl in (240, 365, 540, 730, 1095, 1460, 2190):
        grid = [next(r for r in records if r["half_life"] == hl and r["lam"] == lam
                     and not r["home"] and r["championship_league"]) for lam in (2, 4, 8, 16, 32, 64)]
        lines.append(f"| {hl} 天 | " + " | ".join(f"{r['log_loss']:.4f}" for r in grid) + " |")
    lines += ["", "## 变体", "", "| 主场项 | 冠军联赛训练数据 | 对数损失 | Brier | 命中率 |", "|---|---|---:|---:|---:|"]
    for r in records:
        if r["half_life"] == best["half_life"] and r["lam"] == best["lam"]:
            lines.append(f"| {r['home']} | {r['championship_league']} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['accuracy']:.1%} |")
    lines += ["", "## 解释与限制", "",
              "参数选择和成绩报告使用同一批回测，不能当成独立测试集成绩；微小差异尚未验证统计显著性。",
              "回测使用点估计并在每站开赛前冻结实力；实际 forecast 会对参数不确定性取平均，并可在轮间更新数据。",
              "日期是解析器给出的近似日期，不能据此还原一站赛事内部的逐场顺序。", "",
              "少样本球员的局数、联合后验排名区间和加强收缩的分组回测，见 [少样本诊断](small_sample.md)。",
              "完整实力榜及排名概率统一保存在 [ratings.json](ratings.json)。", ""]
    (ROOT / OUTPUTS[1]).write_text("\n".join(lines))
    import numpy, scipy
    manifest = {"schema_version": 1, "data": stats,
                "backtest_since": "2025-07-01", "configurations": len(records),
                "best": {k: v for k, v in best.items() if not k.startswith("_")},
                "environment": {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__},
                "sha256": hashes(INPUTS + OUTPUTS)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Updated backtest, ratings, small-sample audit, summary and manifest in evaluations/.")


if __name__ == "__main__":
    main()
