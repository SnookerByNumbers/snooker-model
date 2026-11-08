"""Fetch an auditable snooker.org snapshot using an approved X-Requested-By value.

python -I src/fetch_snooker_org.py raw/snooker_org.json 2023 2024 2025 2026
Set SNOOKER_ORG_REQUESTED_BY in the environment; never put it in the snapshot.
Season 2023 means 2023/24. See docs/数据说明.md for access and usage permissions.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import requests

API = "https://api.snooker.org/"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Client:
    def __init__(self, requested_by, interval=1.0, session=None):
        if not requested_by or not requested_by.strip():
            raise ValueError("Set SNOOKER_ORG_REQUESTED_BY to the value approved by snooker.org; "
                             "request access from webmaster@snooker.org first.")
        if "\r" in requested_by or "\n" in requested_by:
            raise ValueError("Invalid X-Requested-By value")
        self.session = session or requests.Session()
        self.session.headers.update({"X-Requested-By": requested_by.strip(),
                                     "User-Agent": "SnookerByNumbers/1.0 "
                                     "(https://github.com/SnookerByNumbers/snooker-model)"})
        self.interval = max(1.0, interval)
        self.last_request = None
        self.requests = []

    def get(self, **params):
        for attempt in range(3):
            if self.last_request is not None:
                time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.session.get(API, params=params, timeout=30, allow_redirects=False)
            except requests.RequestException:
                # Do not include request headers/credentials in exception output.
                if attempt == 2:
                    raise RuntimeError("snooker.org network request failed after 3 attempts") from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code in (401, 403):
                raise RuntimeError("snooker.org rejected access; verify the approved X-Requested-By value")
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 2:
                    raise RuntimeError(f"snooker.org HTTP {response.status_code} after 3 attempts")
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit() and int(retry_after) > 60:
                    raise RuntimeError("snooker.org requested a long retry delay; retry the command later")
                time.sleep(max(2 ** attempt, int(retry_after) if retry_after.isdigit() else 0))
                continue
            if response.status_code != 200:
                raise RuntimeError(f"Unexpected snooker.org HTTP {response.status_code}")
            try:
                rows = response.json()
            except ValueError:
                raise RuntimeError("snooker.org returned invalid JSON") from None
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise RuntimeError("Expected a JSON array of objects from snooker.org")
            self.requests.append({"url": response.url, "fetched_at": now(), "records": len(rows)})
            return rows


def fetch(client, seasons, event_ids=None, catalog_only=False):
    events = {}
    for season in sorted(set(seasons)):
        for event in client.get(t=5, s=season, tr="main"):
            events[str(event["ID"])] = event
    selected = set(map(str, event_ids)) if event_ids else set(events)
    if selected - events.keys():
        raise ValueError("Requested event IDs not found in these seasons: " + ", ".join(sorted(selected - events.keys())))
    players, matches, rounds = {}, [], []
    if not catalog_only:
        for event_id in sorted(selected, key=int):
            event_matches = client.get(t=6, e=event_id)
            if any(str(m.get("EventID")) != event_id for m in event_matches):
                raise ValueError("API returned matches from an unexpected event")
            matches.extend(event_matches)
            rounds.extend(client.get(t=12, e=event_id))
            for player in client.get(t=9, e=event_id):
                players[str(player["ID"])] = player
        missing = {str(m[k]) for m in matches for k in ("Player1ID", "Player2ID")
                   if m.get(k, 0) > 0} - players.keys()
        for player_id in sorted(missing, key=int):
            for player in client.get(p=player_id):
                players[str(player["ID"])] = player
    return {"schema_version": 1, "source": API, "fetched_at": now(),
            "catalog_only": catalog_only, "seasons": sorted(set(seasons)),
            "selected_event_ids": sorted(selected, key=int), "events": events,
            "players": players, "rounds": rounds, "matches": matches,
            "requests": client.requests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("seasons", type=int, nargs="+")
    parser.add_argument("--events", type=int, nargs="+", help="only download these event IDs")
    parser.add_argument("--catalog-only", action="store_true", help="download the event catalog for mapping")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between requests (minimum 1)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Snapshot already exists; use a new filename to preserve provenance")
    try:
        client = Client(os.environ.get("SNOOKER_ORG_REQUESTED_BY"), interval=args.interval)
        snapshot = fetch(client, args.seasons, args.events, args.catalog_only)
    except (ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # No partial snapshot is written when a request fails. Exclusive creation preserves older snapshots.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved {len(snapshot['matches'])} matches to {args.output}")


if __name__ == "__main__":
    main()
