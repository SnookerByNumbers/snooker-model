"""Century breaks per player per event, from the "Century breaks" section of each Wikipedia event page.

Usage: python -I parse_centuries.py RAW_DIR PROCESSED_DIR

Reads RAW_DIR/events.json (fetch_wiki.py) and PROCESSED_DIR/matches.csv (for player ids), writes
PROCESSED_DIR/centuries.csv with one row per century: event (named as in matches.csv), stage (the subsection, e.g. "Main stage centuries";
"" when the page has no subsections), player, break.

Checks: where a subsection states "A total of N century breaks", the parsed count must equal N, otherwise the
script stops. Player names are mapped to matches.csv ids (link target without "(snooker player)", or the plain
name for unlinked amateurs); a name that matches no id is kept as written and reported.
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_matches import SKIP, calendar, drop_templates, event_name, expand_pages, link_parts, name_key, norm_pid, plain  # noqa: E402

LINE_SPLIT = re.compile(r"\{\{\s*ndash\s*\}\}|–")


def section(text):
    m = re.search(r"^==\s*Century breaks\s*==", text, re.M)
    if not m:
        return None
    end = re.search(r"^==[^=]", text[m.end():], re.M)
    return text[m.end(): m.end() + end.start()] if end else text[m.end():]


def breaks(left):
    left = re.sub(r"<small>.*?</small>", " ", left)  # group markers like (5) or (W) in the Championship League
    left = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", left).replace("'''", "")
    left = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", " ", left)
    vals = [int(v) for v in re.findall(r"\d+", left)]
    if any(not 100 <= v <= 155 for v in vals):  # 155 = highest possible with a free ball
        raise SystemExit(f"unexpected break value in: {left!r}")
    return vals


def player(right, ids):
    right = right.replace("\xa0", " ").replace("&nbsp;", " ")
    lp = link_parts(right)
    name = norm_pid(lp[0]) if lp else norm_pid(plain(re.sub(r"<ref.*", "", right)))
    if name not in ids and lp and norm_pid(lp[1]) in ids:
        name = norm_pid(lp[1])
    if name not in ids:  # a spelling variant matches.csv merged ("Lü Chenwei" -> "Lu Chenwei")
        name = BY_KEY.get(name_key(name), name)
    return name


BY_KEY = {}  # name_key -> matches.csv id
LENIENT = "--lenient" in sys.argv  # history data: skip and list what cannot be read instead of stopping
skipped, mismatched = [], []


def main():
    raw, proc = Path(sys.argv[1]), Path(sys.argv[2])
    pages = expand_pages(json.loads((raw / "events.json").read_text()))
    ids = set()
    for r in csv.DictReader(open(proc / "matches.csv")):
        ids.update((r["p1"], r["p2"]))
    BY_KEY.update({name_key(i): i for i in ids})
    meta = calendar(raw)
    rows, unknown, checked = [], set(), 0
    for title, page in sorted(pages.items()):
        if SKIP.search(title):
            continue
        sec = section(page["content"])
        if sec is None:
            continue
        event = event_name(title, page, meta) or title  # the same name as in matches.csv
        # a reference can span lines, and its own bullet lines are not breaks (2019–20 Championship League:
        # "<ref name=century>*Group 1 - {{cite web ...}}\n*Group 2 - ...</ref>")
        sec = re.sub(r"<ref[^>]*?/>", " ", sec)
        sec = re.sub(r"<ref[^>]*>.*?</ref>", " ", sec, flags=re.S)
        if title.endswith(" qualifying"):  # matches.csv files these under the main event; centuries must follow
            raise SystemExit(f"{title}: a separate qualifying page with centuries - map it to its main event first")
        parts = re.split(r"^===\s*(.*?)\s*===\s*$", sec, flags=re.M)
        for stage, body in [("", parts[0])] + list(zip(parts[1::2], parts[2::2])):
            n = 0
            for line in body.splitlines():
                if not line.startswith("*"):
                    continue
                sp = LINE_SPLIT.split(line, maxsplit=1)
                if len(sp) < 2:  # older pages: "* 110 Michael Collumb", "* [[Maximum break|147]], 142 [[John Rea]]"
                    bare = re.sub(r"\[\[(?:[^\]|]*\|)?(\d{3})\]\]", r"\1", line)
                    m = re.match(r"\*\s*((?:'*\d{3}'*\s*,?\s*)+)\s+(\S.*)$", bare)
                    sp = [m.group(1), m.group(2)] if m else sp
                if len(sp) < 2:
                    if LENIENT:
                        skipped.append((title, line))
                        continue
                    raise SystemExit(f"{title}: cannot split {line!r}")
                p = player(sp[1], ids)
                if p not in ids:
                    unknown.add(p)
                try:
                    vals = breaks(sp[0].lstrip("*"))
                except SystemExit:
                    if not LENIENT:
                        raise
                    skipped.append((title, line))
                    continue
                for b in vals:
                    rows.append([event, stage, p, b])
                    n += 1
            # the stated total, once footnotes are out of the way ("A total of 173{{efn|per group ...}} [[century breaks]]")
            m = re.search(r"total of (\d+)\s*\[\[century break", drop_templates(body, ["efn", "refn", "sfn"]))
            if m:
                checked += 1
                if int(m.group(1)) != n:
                    if not LENIENT:
                        raise SystemExit(f"{title} / {stage}: parsed {n} centuries, page says {m.group(1)}")
                    mismatched.append((title, stage, n, int(m.group(1))))
    with open(proc / "centuries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "stage", "player", "break"])
        w.writerows(rows)
    print(f"{len(rows)} centuries in {len({r[0] for r in rows})} events; {checked} stated totals match")
    if unknown:
        print("names with no match in matches.csv:", ", ".join(sorted(unknown)))
    if LENIENT:
        (proc / "centuries_skipped.json").write_text(json.dumps({"unreadable_lines": skipped, "total_mismatch":
                                                                 mismatched}, ensure_ascii=False, indent=1))
        print(f"lenient: {len(skipped)} lines skipped, {len(mismatched)} sections whose total does not match "
              f"-> {proc / 'centuries_skipped.json'}")


if __name__ == "__main__":
    main()
