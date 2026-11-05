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

API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks API clients to identify themselves with a way to reach them; set WIKI_USER_AGENT to your own
# "name/version (contact URL or e-mail)" when you run this.
HEADERS = {"User-Agent": os.environ.get("WIKI_USER_AGENT",
                                        "SnookerByNumbers-model/1.0 (https://github.com/SnookerByNumbers)")}
SKIP = re.compile(r"Shoot Out|Mixed Doubles|Nations|Six-red|Seniors|Women|World Cup|Q School|Under-21", re.I)


def get_pages(titles):
    """Fetch current wikitext for titles (batched); returns {resolved title: (content, timestamp)}."""
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
            for p in d["query"]["pages"]:
                if p.get("missing") or "revisions" not in p:
                    continue
                rev = p["revisions"][0]
                if "content" in rev["slots"]["main"]:
                    out[p["title"]] = (rev["slots"]["main"]["content"], rev["timestamp"])
            if "continue" not in d:
                break
            params.update(d["continue"])
        time.sleep(0.5)
    return out


def wst_event_titles(season_text):
    """Event page titles linked from the 'World Snooker Tour' calendar table of a season page."""
    m = re.search(r"===\s*World Snooker Tour\s*===(.*?)\n===?[^=]", season_text, re.S)
    section = m.group(1) if m else ""
    titles = []
    for target in re.findall(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", section):
        target = target.strip()
        if re.match(r"20\d\d ", target) and not SKIP.search(target) and target not in titles:
            titles.append(target)
    return titles


def main():
    out_dir = Path(sys.argv[1])
    seasons = [f"{s[:4]}–{s[-2:]} snooker season" for s in sys.argv[2:]]
    season_pages = get_pages(seasons)
    events = []
    for title, (text, _) in season_pages.items():
        found = wst_event_titles(text)
        print(f"{title}: {len(found)} events")
        events += [t for t in found if t not in events]
    pages = get_pages(events)
    quals = get_pages([f"{t} qualifying" for t in events])
    print(f"fetched {len(pages)} event pages, {len(quals)} qualifying pages")
    pages.update(quals)
    (out_dir / "seasons.json").write_text(json.dumps({k: v[0] for k, v in season_pages.items()}, ensure_ascii=False))
    (out_dir / "events.json").write_text(json.dumps(
        {k: {"content": v[0], "timestamp": v[1]} for k, v in pages.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
