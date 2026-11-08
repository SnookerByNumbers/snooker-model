"""Audit, then optionally enrich existing Wikipedia matches with snooker.org metadata.

python -I src/enrich_snooker_org.py SNAPSHOT DATA_DIR MAPPING --report output/snooker_org_audit.json
Add --apply to update matches.csv after a conflict-free audit. No results are added or changed.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_matches import norm_pid

FIELDS = ["date_wikipedia", "date_source", "snooker_org_key", "snooker_org_event_id",
          "snooker_org_p1_id", "snooker_org_p2_id", "snooker_org_round", "snooker_org_round_name",
          "snooker_org_scheduled_at", "snooker_org_started_at", "snooker_org_ended_at",
          "snooker_org_estimated", "snooker_org_source_url", "snooker_org_fetched_at"]


def name_key(name):
    return " ".join(unicodedata.normalize("NFKC", norm_pid(name)).replace("’", "'").casefold().split())


def timestamp(value):
    if not value:
        return None
    try:
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if value.tzinfo is None or value.year < 1900:
        return None
    return value.astimezone(timezone.utc)


def match_key(match):
    # The API's ID is a revision identifier, NOT a stable match identifier.
    return ":".join(str(int(match[k])) for k in ("EventID", "Round", "Number"))


def enrich(rows, local_events, snapshot, mapping):
    if snapshot.get("schema_version") != 1 or snapshot.get("source") != "https://api.snooker.org/":
        raise ValueError("Not a supported snooker.org snapshot")
    if snapshot.get("catalog_only"):
        raise ValueError("Catalog-only snapshots contain no match results")
    fetched_at = timestamp(snapshot.get("fetched_at"))
    if fetched_at is None:
        raise ValueError("Snapshot needs a timezone-aware fetched_at")
    if mapping.get("schema_version") != 1:
        raise ValueError("Unsupported mapping schema")
    event_map, player_map = mapping["events"], mapping["players"]
    names = {r[k] for r in rows for k in ("p1", "p2")}
    by_name = defaultdict(set)
    for name in names:
        by_name[name_key(name)].add(name)
    for target in event_map.values():
        if target["event"] not in local_events or target["stage"] not in ("main", "qualifying"):
            raise ValueError("Event mapping must reference a local event and an explicit main/qualifying stage")
    if any(name not in names for name in player_map.values()):
        raise ValueError("Player mappings must use names already present in matches.csv")

    def player_name(pid):
        if str(pid) in player_map:
            return player_map[str(pid)]
        player = snapshot["players"].get(str(pid))
        if not player or player.get("Type", 1) != 1:
            return None
        # Use the repository's existing aliases; do not invent fuzzy player-name matches.
        parts = [player.get(k, "") or "" for k in ("FirstName", "MiddleName", "LastName")]
        options = set()
        for ordered in (parts, [parts[2], parts[0], parts[1]]):
            options.update(by_name.get(name_key(" ".join(ordered)), ()))
        return next(iter(options)) if len(options) == 1 else None

    resolved_players = {str(pid): player_name(pid) for m in snapshot["matches"]
                        for pid in (m["Player1ID"], m["Player2ID"])}
    ids_per_name = defaultdict(set)
    for pid, name in resolved_players.items():
        if name is not None:
            ids_per_name[name].add(pid)
    ambiguous_names = {name for name, ids in ids_per_name.items() if len(ids) > 1}
    local = defaultdict(list)
    for i, row in enumerate(rows):
        local[(row["event"], row.get("stage", "main"), frozenset((row["p1"], row["p2"])))].append(i)
    rounds = {(str(r["EventID"]), str(r["Round"])): r.get("RoundName", "") for r in snapshot["rounds"]}
    keys = Counter(match_key(m) for m in snapshot["matches"])
    proposals, issues, skipped = defaultdict(list), [], Counter()

    def issue(key, reason, **details):
        issues.append({"key": key, "reason": reason, **details})

    for match in snapshot["matches"]:
        key = match_key(match)
        if keys[key] != 1:
            issue(key, "duplicate_source_key")
            continue
        if (match.get("Status") != 3 or match.get("Type") != 1 or
                match.get("Walkover1") or match.get("Walkover2") or match.get("Round") == 17):
            skipped["unfinished_walkover_or_nonstandard"] += 1
            continue
        eid = str(match["EventID"])
        if eid not in event_map:
            skipped["unmapped_event"] += 1
            continue
        p1, p2 = resolved_players[str(match["Player1ID"])], resolved_players[str(match["Player2ID"])]
        if p1 is None or p2 is None or p1 == p2 or p1 in ambiguous_names or p2 in ambiguous_names:
            issue(key, "unresolved_player", player_ids=[match["Player1ID"], match["Player2ID"]])
            continue
        scores = (match.get("Score1"), match.get("Score2"))
        if any(type(s) is not int or s < 0 for s in scores) or sum(scores) == 0:
            issue(key, "invalid_score")
            continue
        target = event_map[eid]
        candidates = local.get((target["event"], target["stage"], frozenset((p1, p2))), [])
        if not candidates:
            skipped["no_local_match"] += 1
            continue
        # Reuse a previously confirmed identity; otherwise require a unique score match.
        identified = [i for i in candidates if rows[i].get("snooker_org_key") == key]
        exact = []
        for i in identified or candidates:
            r = rows[i]
            a, b = scores if r["p1"] == p1 else scores[::-1]
            if (int(r["f1"]), int(r["f2"])) == (a, b):
                exact.append(i)
        if len(exact) != 1:
            issue(key, "ambiguous_match" if len(exact) > 1 else "score_conflict", rows=[i + 2 for i in candidates])
            continue
        i = exact[0]
        old_key = rows[i].get("snooker_org_key")
        if old_key and old_key != key:
            issue(key, "existing_identity_conflict", row=i + 2)
            continue
        old_fetch = timestamp(rows[i].get("snooker_org_fetched_at"))
        if old_fetch and old_fetch > fetched_at:
            issue(key, "older_snapshot", row=i + 2)
            continue
        ids = (match["Player1ID"], match["Player2ID"])
        if rows[i]["p1"] != p1:
            ids = ids[::-1]
        extra = {"snooker_org_key": key, "snooker_org_event_id": eid,
                 "snooker_org_p1_id": str(ids[0]), "snooker_org_p2_id": str(ids[1]),
                 "snooker_org_round": str(match["Round"]),
                 "snooker_org_round_name": rounds.get((eid, str(match["Round"])), ""),
                 "snooker_org_scheduled_at": match.get("ScheduledDate") or "",
                 "snooker_org_started_at": match.get("StartDate") or "",
                 "snooker_org_ended_at": match.get("EndDate") or "",
                 "snooker_org_estimated": str(bool(match.get("Estimated"))).lower(),
                 "snooker_org_source_url": f"https://api.snooker.org/?e={eid}&r={match['Round']}&n={match['Number']}",
                 "snooker_org_fetched_at": snapshot["fetched_at"],
                 "date_wikipedia": rows[i].get("date_wikipedia") or rows[i]["date"],
                 "date_source": "wikipedia",
                 "date": rows[i].get("date_wikipedia") or rows[i]["date"]}
        end, start = timestamp(match.get("EndDate")), timestamp(match.get("StartDate"))
        # Scheduled/started dates do not establish when a final result became available.
        # Missing EndDate keeps the original model date; never infer completion from ModDate.
        if end is not None:
            source_event = snapshot["events"][eid]
            first = min(source_event["StartDate"], local_events[target["event"]]["start_date"])
            last = max(source_event["EndDate"], local_events[target["event"]]["end_date"])
            low = datetime.fromisoformat(first).replace(tzinfo=timezone.utc) - timedelta(days=1)
            high = datetime.fromisoformat(last).replace(tzinfo=timezone.utc) + timedelta(days=2)
            if end > fetched_at or (start is not None and end < start) or not low <= end < high:
                issue(key, "invalid_completion_time", row=i + 2)
                continue
            extra.update(date=end.date().isoformat(), date_source="snooker.org:EndDate:UTC")
        else:
            skipped["matched_without_completion_time"] += 1
        proposals[i].append((key, extra))

    result, changes = [dict(r) for r in rows], []
    for i, options in proposals.items():
        if len(options) != 1:
            issue(",".join(k for k, _ in options), "multiple_source_matches_for_local_row", row=i + 2)
            continue
        key, extra = options[0]
        result[i].update(extra)
        delta = {k: {"before": rows[i].get(k, ""), "after": v}
                 for k, v in extra.items() if rows[i].get(k, "") != v}
        changes.append({"row": i + 2, "key": key, "changes": delta})
    report = {"source": snapshot["source"], "fetched_at": snapshot["fetched_at"],
              "local_rows": len(rows), "matched_rows": len(changes),
              "date_changes": sum("date" in c["changes"] for c in changes),
              "issues": issues, "skipped": dict(skipped), "changes": changes,
              "unmapped_event_ids": sorted({str(m["EventID"]) for m in snapshot["matches"]} - event_map.keys(), key=int)}
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    matches_path = args.data_dir / "matches.csv"
    protected = [matches_path, args.data_dir / "events.csv", args.snapshot, args.mapping]
    if args.report.resolve() in {p.resolve() for p in protected}:
        parser.error("Report must not overwrite an input file")
    before = matches_path.read_bytes()
    reader = csv.DictReader(io.StringIO(before.decode("utf-8")))
    fields, rows = reader.fieldnames, list(reader)
    with (args.data_dir / "events.csv").open(encoding="utf-8") as stream:
        events = {r["event"]: r for r in csv.DictReader(stream)}
    snapshot_bytes, mapping_bytes = args.snapshot.read_bytes(), args.mapping.read_bytes()
    try:
        result, report = enrich(rows, events, json.loads(snapshot_bytes), json.loads(mapping_bytes))
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot enrich: {error}\n")
    report["sha256"] = {"matches_before": hashlib.sha256(before).hexdigest(),
                        "snapshot": hashlib.sha256(snapshot_bytes).hexdigest(),
                        "mapping": hashlib.sha256(mapping_bytes).hexdigest()}
    report["applied"] = False
    if args.apply and report["matched_rows"] and not report["issues"]:
        if matches_path.read_bytes() != before:
            parser.exit(1, "matches.csv changed during the audit; rerun before applying\n")
        fd, temporary = tempfile.mkstemp(dir=args.data_dir, prefix=".matches-", suffix=".csv")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields + [f for f in FIELDS if f not in fields])
                writer.writeheader()
                writer.writerows(result)
            os.replace(temporary, matches_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        report["applied"] = True
        report["sha256"]["matches_after"] = hashlib.sha256(matches_path.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Matched {report['matched_rows']}/{len(rows)} rows; {report['date_changes']} date changes; "
          f"{len(report['issues'])} issues; applied={report['applied']}. Audit: {args.report}")
    if report["issues"] or (args.apply and not report["matched_rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
