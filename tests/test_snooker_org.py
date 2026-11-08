"""Source reconciliation must never turn ambiguous identities or future results into training data."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from enrich_snooker_org import enrich
from fetch_snooker_org import Client, fetch


def fixture():
    # Synthetic records, shaped like the public API documentation; not downloaded match data.
    rows = [{"date": "2026-10-10", "event": "Synthetic Open", "country": "England",
             "p1": "Alpha Player", "p2": "Beta Player", "f1": "6", "f2": "3",
             "nat1": "ENG", "nat2": "ENG", "stage": "main"}]
    events = {"Synthetic Open": {"event": "Synthetic Open", "start_date": "2026-10-01",
                                  "end_date": "2026-10-10", "country": "England"}}
    snapshot = {"schema_version": 1, "source": "https://api.snooker.org/",
                "fetched_at": "2026-10-11T12:00:00Z", "catalog_only": False,
                "events": {"101": {"ID": 101, "StartDate": "2026-10-01", "EndDate": "2026-10-10"}},
                "players": {"11": {"ID": 11, "FirstName": "Alpha", "LastName": "Player", "Type": 1},
                            "12": {"ID": 12, "FirstName": "Beta", "LastName": "Player", "Type": 1}},
                "rounds": [{"EventID": 101, "Round": 7, "RoundName": "Round 1"}],
                "matches": [{"ID": 999, "EventID": 101, "Round": 7, "Number": 4,
                             "Player1ID": 11, "Player2ID": 12, "Score1": 6, "Score2": 3,
                             "Status": 3, "Type": 1, "Walkover1": False, "Walkover2": False,
                             "ScheduledDate": "2026-10-02T12:00:00Z", "Estimated": False,
                             "StartDate": "2026-10-02T12:10:00Z", "EndDate": "2026-10-02T15:00:00Z"}]}
    mapping = {"schema_version": 1, "events": {"101": {"event": "Synthetic Open", "stage": "main"}}, "players": {}}
    return rows, events, snapshot, mapping


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.events, self.snapshot, self.mapping = fixture()
        self.match = self.snapshot["matches"][0]

    def run_enrich(self):
        return enrich(self.rows, self.events, self.snapshot, self.mapping)

    def test_completed_match_and_reversed_players(self):
        self.match.update(Player1ID=12, Player2ID=11, Score1=3, Score2=6)
        result, report = self.run_enrich()
        self.assertEqual(report["issues"], [])
        self.assertEqual(result[0]["snooker_org_p1_id"], "11")
        self.assertEqual(result[0]["snooker_org_key"], "101:7:4")
        self.assertEqual(result[0]["date"], "2026-10-02")
        self.assertEqual(result[0]["date_wikipedia"], "2026-10-10")
        self.assertEqual(result[0]["snooker_org_round_name"], "Round 1")
        self.assertEqual(self.rows[0]["date"], "2026-10-10")
        for k in ("f1", "f2", "p1", "p2", "stage", "nat1", "nat2"):
            self.assertEqual(result[0][k], self.rows[0][k])

    def test_scheduled_date_is_never_used_as_completion(self):
        self.match.update(EndDate="", Estimated=True)
        result, report = self.run_enrich()
        self.assertEqual(result[0]["date"], self.rows[0]["date"])
        self.assertEqual(result[0]["date_source"], "wikipedia")
        self.assertEqual(report["date_changes"], 0)

    def test_utc_completion_and_multiday_matches(self):
        self.match["EndDate"] = "2026-10-04T00:30:00+08:00"
        result, _ = self.run_enrich()
        self.assertEqual(result[0]["date"], "2026-10-03")

    def test_future_or_impossible_completion_is_rejected(self):
        for end in ("2026-10-20T12:00:00Z", "2026-10-01T12:00:00Z", "2025-10-02T15:00:00Z"):
            with self.subTest(end=end):
                self.match["EndDate"] = end
                result, report = self.run_enrich()
                self.assertEqual(result, self.rows)
                self.assertEqual(report["issues"][0]["reason"], "invalid_completion_time")

    def test_live_walkover_team_and_champion_rows_are_excluded(self):
        for change in ({"Status": 1}, {"Status": 2}, {"Walkover1": True}, {"Type": 3}, {"Round": 17}):
            with self.subTest(change=change):
                self.snapshot["matches"] = [{**self.match, **change}]
                result, report = self.run_enrich()
                self.assertEqual(result, self.rows)
                self.assertEqual(report["matched_rows"], 0)

    def test_score_conflict_does_not_overwrite_wikipedia(self):
        self.match["Score2"] = 4
        result, report = self.run_enrich()
        self.assertEqual(result, self.rows)
        self.assertEqual(report["issues"][0]["reason"], "score_conflict")

    def test_repeated_identical_matchups_are_ambiguous(self):
        self.rows.append(dict(self.rows[0]))
        result, report = self.run_enrich()
        self.assertEqual(result, self.rows)
        self.assertEqual(report["issues"][0]["reason"], "ambiguous_match")

    def test_two_source_matches_cannot_claim_one_local_row(self):
        self.snapshot["matches"].append({**self.match, "Number": 5})
        result, report = self.run_enrich()
        self.assertEqual(result, self.rows)
        self.assertEqual(report["issues"][0]["reason"], "multiple_source_matches_for_local_row")

    def test_duplicate_source_keys_rejected_even_if_revision_id_differs(self):
        self.snapshot["matches"].append({**self.match, "ID": 1000})
        result, report = self.run_enrich()
        self.assertEqual(result, self.rows)
        self.assertEqual(report["issues"][0]["reason"], "duplicate_source_key")

    def test_repeated_run_is_idempotent_and_revision_id_ignored(self):
        self.rows, _ = self.run_enrich()
        self.match["ID"] += 1
        result, report = self.run_enrich()
        self.assertEqual(result, self.rows)
        self.assertEqual(report["changes"][0]["changes"], {})

    def test_older_snapshot_cannot_replace_newer_enrichment(self):
        self.rows, _ = self.run_enrich()
        self.snapshot["fetched_at"] = "2026-10-10T12:00:00Z"
        result, report = self.run_enrich()
        self.assertEqual(result, self.rows)
        self.assertEqual(report["issues"][0]["reason"], "older_snapshot")

    def test_unknown_player_requires_mapping(self):
        self.snapshot["players"]["11"]["FirstName"] = "Different"
        _, report = self.run_enrich()
        self.assertEqual(report["issues"][0]["reason"], "unresolved_player")
        self.mapping["players"]["11"] = "Alpha Player"
        _, report = self.run_enrich()
        self.assertEqual(report["matched_rows"], 1)

    def test_two_player_ids_with_same_name_are_not_silently_merged(self):
        self.snapshot["players"]["13"] = {**self.snapshot["players"]["11"], "ID": 13}
        self.snapshot["matches"].append({**self.match, "Number": 5, "Player1ID": 13})
        result, report = self.run_enrich()
        self.assertEqual(result, self.rows)
        self.assertEqual(report["matched_rows"], 0)

    def test_event_and_stage_are_explicit(self):
        self.mapping["events"] = {}
        result, report = self.run_enrich()
        self.assertEqual(result, self.rows)
        self.assertEqual(report["unmapped_event_ids"], ["101"])
        self.mapping["events"] = {"101": {"event": "Synthetic Open", "stage": "qualifying"}}
        _, report = self.run_enrich()
        self.assertEqual(report["matched_rows"], 0)

    def test_cli_audits_before_apply_and_preserves_input_on_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename, rows in (("matches.csv", self.rows), ("events.csv", list(self.events.values()))):
                with (root / filename).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            for name, data in (("snapshot.json", self.snapshot), ("mapping.json", self.mapping)):
                (root / name).write_text(json.dumps(data))
            command = [sys.executable, "-I", str(ROOT / "src/enrich_snooker_org.py"),
                       str(root / "snapshot.json"), str(root), str(root / "mapping.json"),
                       "--report", str(root / "audit.json")]
            before = (root / "matches.csv").read_bytes()
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual((root / "matches.csv").read_bytes(), before)
            subprocess.run(command + ["--apply"], check=True, capture_output=True)
            after = (root / "matches.csv").read_bytes()
            self.assertNotEqual(before, after)
            self.match["Score2"] = 4
            (root / "snapshot.json").write_text(json.dumps(self.snapshot))
            failure = subprocess.run(command + ["--apply"], capture_output=True)
            self.assertEqual(failure.returncode, 1)
            self.assertEqual((root / "matches.csv").read_bytes(), after)
            self.assertFalse(json.loads((root / "audit.json").read_text())["applied"])


class FetchTests(unittest.TestCase):
    def test_missing_approval_fails_before_network(self):
        session = Mock()
        with self.assertRaisesRegex(ValueError, "approved"):
            Client("", session=session)
        session.get.assert_not_called()

    def test_access_denied_is_not_retried(self):
        session = Mock(headers={})
        session.get.return_value.status_code = 403
        with self.assertRaisesRegex(RuntimeError, "rejected access"):
            Client("synthetic-test-value", session=session).get(t=5, s=2026)
        self.assertEqual(session.get.call_count, 1)

    @patch("fetch_snooker_org.time.sleep")
    def test_retry_then_success_and_no_header_in_provenance(self, sleep):
        session = Mock(headers={})
        retry = Mock(status_code=429, headers={"Retry-After": "1"})
        ok = Mock(status_code=200, url="https://api.snooker.org/?t=5&s=2026")
        ok.json.return_value = [{"ID": 101}]
        session.get.side_effect = [retry, ok]
        client = Client("synthetic-secret", session=session)
        self.assertEqual(client.get(t=5, s=2026), [{"ID": 101}])
        self.assertNotIn("synthetic-secret", json.dumps(client.requests))
        self.assertFalse(session.get.call_args.kwargs["allow_redirects"])

    def test_fetch_season_and_event_filter(self):
        _, _, snapshot, _ = fixture()
        client = Mock(requests=[])
        client.get.side_effect = [list(snapshot["events"].values()), snapshot["matches"],
                                  snapshot["rounds"], list(snapshot["players"].values())]
        result = fetch(client, [2026], [101])
        self.assertEqual(result["matches"], snapshot["matches"])
        self.assertEqual(result["selected_event_ids"], ["101"])
        self.assertEqual(client.get.call_count, 4)

    def test_invalid_json_schema_rejected(self):
        session = Mock(headers={})
        session.get.return_value.status_code = 200
        session.get.return_value.json.return_value = {"error": "bad request"}
        with self.assertRaisesRegex(RuntimeError, "array"):
            Client("synthetic-test-value", session=session).get(t=5, s=2026)


class BacktestStageTests(unittest.TestCase):
    def test_held_over_qualifier_is_not_a_main_stage_test_match(self):
        from audit_samples import match_support
        from model import Data, backtest
        rows, events, _, _ = fixture()
        rows.append({**rows[0], "stage": "qualifying", "date": "2026-10-01", "f1": "0", "f2": "6"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename, data in (("matches.csv", rows), ("events.csv", list(events.values()))):
                with (root / filename).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                    writer.writeheader()
                    writer.writerows(data)
            d = Data(root)
            with patch("builtins.print"):
                results = backtest(d, [(1095, 8, False, True)], "2025-07-01")
            support = match_support(d, 1095, True)
            self.assertEqual(results[0]["n_matches"], 1)
            self.assertEqual(results[0]["_y"], [1.0])
            self.assertEqual([r["y"] for r in support], [1])


if __name__ == "__main__":
    unittest.main()
