"""Download Wikipedia wikitext for snooker season pages and every World Snooker Tour event they list.

Usage: python -I fetch_wiki.py OUT_DIR SEASON [SEASON ...]   (season like 2025-26)

Writes OUT_DIR/seasons.json (season title -> wikitext) and OUT_DIR/events.json
(event title -> {"content": wikitext, "timestamp": revision time}).
Qualifying pages ("<event> qualifying") are fetched too when they exist.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I leaves the script's folder off sys.path
from parse_matches import SKIP, season_events  # noqa: E402

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": os.environ.get("WIKI_USER_AGENT",
    "SnookerByNumbers-model/1.0 (https://github.com/SnookerByNumbers/snooker-model)")}


def get_pages(titles):
    """Fetch current wikitext for titles (batched); returns {resolved title: (content, timestamp, requested title)}.
    The requested title matters when a calendar links a redirect ("2016 PTC Finals" -> "2016 Players Tour
    Championship Final"): parse_matches.py finds the event's dates under the title the calendar used."""
    out = {}
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        params = {"action": "query", "prop": "revisions", "rvprop": "content|timestamp",
                  "rvslots": "main", "titles": "|".join(batch), "redirects": 1,
                  "format": "json", "formatversion": 2}
        while True:
            r = requests.get(API, params=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            d = r.json()
            back = {}  # resolved title -> the title we asked for
            for step in d["query"].get("normalized", []) + d["query"].get("redirects", []):
                back[step["to"]] = back.get(step["from"], step["from"])
            for p in d["query"]["pages"]:
                if p.get("missing") or "revisions" not in p:
                    continue
                rev = p["revisions"][0]
                if "content" in rev["slots"]["main"]:
                    out[p["title"]] = (rev["slots"]["main"]["content"], rev["timestamp"], back.get(p["title"], p["title"]))
            if "continue" not in d:
                break
            params.update(d["continue"])
        time.sleep(0.5)
    return out


def wst_event_titles(season_title, season_text):
    """Event page titles from the 'World Snooker Tour' calendar table of a season page: each row's tournament,
    read the same way parse_matches.py reads the calendar (so both agree on which events exist), minus SKIP."""
    return [t for t in season_events(season_title, season_text) if not SKIP.search(t)]


def main():
    out_dir = Path(sys.argv[1])
    # "1999-00" -> "1999–2000 snooker season" (the article spells out the year at the century)
    seasons = [f"{s[:4]}–{'2000' if s[:4] == '1999' else s[-2:]} snooker season" for s in sys.argv[2:]]
    season_pages = get_pages(seasons)
    events = []
    for title, (text, *_) in season_pages.items():
        found = wst_event_titles(title, text)
        print(f"{title}: {len(found)} events")
        events += [t for t in found if t not in events]
    pages = get_pages(events)
    quals = get_pages([f"{t} qualifying" for t in events])
    print(f"fetched {len(pages)} event pages, {len(quals)} qualifying pages")
    pages.update(quals)
    (out_dir / "seasons.json").write_text(json.dumps({k: v[0] for k, v in season_pages.items()}, ensure_ascii=False))
    (out_dir / "events.json").write_text(json.dumps(
        {k: {"content": v[0], "timestamp": v[1], "requested": v[2]} for k, v in pages.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
