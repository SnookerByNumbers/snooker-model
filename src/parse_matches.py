"""Parse match results (player, frames won) out of the downloaded Wikipedia wikitext.

Usage: python -I parse_matches.py RAW_DIR OUT_DIR

Reads RAW_DIR/seasons.json and RAW_DIR/events.json, writes OUT_DIR/matches.csv
(one row per completed match) and OUT_DIR/events.csv (event dates and host country).
Handles bracket templates (named RDn-teamNN params and positional params),
bullet-list results ("* A 6–4 B"), which covers qualifying rounds and
Championship League group matches, and the final shown as its own wikitable
("'''Final:''' Best of 19 frames" caption, then a row "player | score | player"):
many pages draw the bracket only up to the semi-finals.
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DROP_TEMPLATES = ("efn", "refn", "cite", "tooltip", "dagger", "abbr", "sfn", "cbignore", "small")
DASH = r"(?:–|—|-|\{\{\s*(?:ndash|nbnd|snd)\s*\}\}|&ndash;|&mdash;|&#821[12];|&#x201[34];)"
NOT_PLAYERS = {"bye", "tbd", "tba", "withdrawn", "w/d", "w/o", "walkover", ""}
SUFFIX = re.compile(r"\s*\((?:snooker player|snooker)\)$")  # as in canonical_ids


def template_end(text, start):
    """Index just past the template that opens at text[start] ('{{')."""
    depth, i = 0, start
    while i < len(text) - 1:
        if text.startswith("{{", i):
            depth, i = depth + 1, i + 2
        elif text.startswith("}}", i):
            depth, i = depth - 1, i + 2
            if depth == 0:
                return i
        else:
            i += 1
    return len(text)


def drop_templates(text, names):
    pat = re.compile(r"\{\{\s*(?:" + "|".join(names) + r")\b", re.I)
    while (m := pat.search(text)):
        text = text[:m.start()] + text[template_end(text, m.start()):]
    return text


def clean(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>/]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"<sup>.*?</sup>", "", text, flags=re.S)
    return drop_templates(text, DROP_TEMPLATES)


def split_top(body):
    """Split a template body on '|' that are not nested inside {{ }} or [[ ]]."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(body):
        two = body[i:i + 2]
        if two in ("{{", "[["):
            depth += 1; cur.append(two); i += 2; continue
        if two in ("}}", "]]"):
            depth -= 1; cur.append(two); i += 2; continue
        if body[i] == "|" and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(body[i])
        i += 1
    parts.append("".join(cur))
    return parts


def link_parts(s):
    m = re.search(r"\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]", s)
    if not m:
        return None
    target = m.group(1).strip()
    target = target[0].upper() + target[1:] if target else target
    return target, (m.group(2) or re.sub(r"\s*\(.*?\)$", "", target)).strip()


def plain(s):
    s = re.sub(r"\{\{[^{}]*\}\}", " ", s)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("'''", "").replace("''", "")
    s = re.sub(r"\(\s*(?:\d+|a|wc|WC|q)\s*\)", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .,:;*†‡")


def parse_player(s):
    """Return (id, display, nationality) for a player cell, or None."""
    s = s.replace("'''", "").replace("''", "")
    m = re.search(r"\{\{\s*flagathlete\s*\|", s, re.I)
    if m:
        inner = split_top(s[m.end():template_end(s, m.start()) - 2])
        name, nat = inner[0], (inner[1].strip() if len(inner) > 1 else "")
        lp = link_parts(name)
        if lp:
            return lp[0], lp[1], nat
        name = plain(name)
        return (name, name, nat) if name.lower() not in NOT_PLAYERS else None
    nat = ""
    m = re.search(r"\{\{\s*flagicon\s*\|\s*([A-Za-z]+)", s, re.I) or re.search(r"\{\{\s*([A-Z]{3})\s*\}\}", s)
    if m:
        nat = m.group(1)
    lp = link_parts(s)
    if lp:
        return lp[0], lp[1], nat
    name = plain(s)
    if not name or name.lower() in NOT_PLAYERS or re.fullmatch(r"[\d\s/()a-z.–-]*", name):
        return None
    return name, name, nat


def parse_score(s):
    s = plain(s).lower().replace(" ", "")
    if re.fullmatch(r"\d{1,2}", s):
        return int(s)
    if s in ("w/o", "wo", "w/d", "wd", "w.o.", "walkover", "ret", "ret."):
        return "wo"
    return None


def bracket_matches(body):
    params = split_top(body)[1:]
    named, positional = defaultdict(dict), []
    for p in params:
        m = re.match(r"\s*RD(\d+)-(team|score)0*(\d+)\s*=(.*)$", p, re.S)
        if m:
            named[int(m.group(1))][(m.group(2), int(m.group(3)))] = m.group(4)
        elif not re.match(r"\s*[\w\- ]+\s*=", p, re.S):
            positional.append(p)
    out = []
    for rnd, d in named.items():
        n = max((k[1] for k in d), default=0)
        for i in range(1, n + 1, 2):
            a, b = parse_player(d.get(("team", i), "")), parse_player(d.get(("team", i + 1), ""))
            sa, sb = parse_score(d.get(("score", i), "")), parse_score(d.get(("score", i + 1), ""))
            if a and b:
                out.append((a, sa, b, sb))
    toks = []
    for p in positional:
        if not p.strip():
            continue
        sc = parse_score(p)
        if sc is not None:
            toks.append(("S", sc))
        elif (pl := parse_player(p)):
            toks.append(("P", pl))
    i = 0
    while i < len(toks):  # P S [S(seed)] P S
        if toks[i][0] == "P" and i + 3 < len(toks) and toks[i + 1][0] == "S":
            j = i + 2
            if toks[j][0] == "S" and j + 2 < len(toks):
                j += 1
            if j + 1 < len(toks) and toks[j][0] == "P" and toks[j + 1][0] == "S":
                out.append((toks[i][1], toks[i + 1][1], toks[j][1], toks[j + 1][1]))
                i = j + 2
                continue
        i += 1
    return out


def list_matches(text):
    out = []
    for line in text.splitlines():
        if not line.lstrip().startswith(("*", "#")):
            continue
        line = line.lstrip("*# ").replace("'''", "").replace("''", "")
        scores = list(re.finditer(r"(?<![\d,£])(\d{1,2})\s*" + DASH + r"\s*(\d{1,2})(?![\d,])", line))
        if len(scores) != 1:
            continue
        m = scores[0]
        a, b = parse_player(line[:m.start()]), parse_player(line[m.end():])
        if a and b:
            out.append((a, int(m.group(1)), b, int(m.group(2))))
    return out


def cell(c):
    """Content of a wikitable cell, its leading '|' already gone ('width="260" align="right" | [[A]]' -> '[[A]]')."""
    m = re.match(r"\s*[^\[{|\n]*=[^\[{|\n]*\|(?!\|)", c)  # attributes: an "=" and no link or template before the bar
    return c[m.end():] if m else c


FINAL_CAPTION = re.compile(r"^[|!][^\n]*?'''\s*Final\s*(?::\s*'''|'''\s*:?)", re.M | re.I)


def final_tables(text):
    """The match under each final table's caption (bold "Final:" / "Final" on a table row, e.g. "'''Final:''' Best
    of 19 frames"): the first row of three cells player | score | player, one cell per line or "| a || b || c".
    A caption with no such row is reported, so a new layout is not dropped silently."""
    out = []
    for m in FINAL_CAPTION.finditer(text):
        end = text.find("\n|}", m.end())
        body = text[m.end(): end if end >= 0 else m.end() + 3000]
        for row in re.split(r"\n\|-[^\n]*", body)[1:]:
            cells = [cell(x) for line in row.strip("\n").split("\n")
                     if line.startswith("|") and not line.startswith(("|-", "|}", "|+")) for x in line[1:].split("||")]
            if len(cells) != 3:
                continue
            cells = [re.sub(r"</?b>", "", x) for x in cells]
            sc = re.fullmatch(r"\s*(\d{1,2})\s*" + DASH + r"\s*(\d{1,2})\s*", cells[1].replace("'''", ""))
            a, b = parse_player(cells[0]), parse_player(cells[2])
            if sc and a and b:
                out.append((a, int(sc.group(1)), b, int(sc.group(2))))
                break
        else:
            print("final table without a 'player | score | player' row:",
                  text[m.start():m.start() + 120].replace("\n", " "), file=sys.stderr)
    return out


def section_matches(text):
    out, rest, pos = [], [], 0  # out: bracket results
    for m in re.finditer(r"\{\{\s*(?:[^|{}]*Bracket[^|{}]*|#invoke:\s*RoundN[^|{}]*)\|", text):
        if m.start() < pos:
            continue
        end = template_end(text, m.start())
        rest.append(text[pos:m.start()])
        out += bracket_matches(text[m.start() + 2:end - 2])
        pos = end
    rest.append(text[pos:])
    rest = "".join(rest)
    # source of each result: "bracket", "list", or "final" (a final read from its own table, which only fills in a
    # final the page shows nowhere else)
    return [(*m, "bracket") for m in out] + [(*m, "list") for m in list_matches(rest)] + \
        [(*m, "final") for m in final_tables(rest)]


def qualifying_date(text, end_date):
    """Last day of qualifying from the lead ('Qualifiers took place from 17 to 19 September'), or None."""
    m = re.search(r"Qualif\w*\s+(?:took|take|were|was|are|is|will)[^.]*?(\d{1,2})\s+([A-Z][a-z]+)", text)
    if not m or m.group(2)[:3].lower() not in MONTHS:
        return None
    day, mon = int(m.group(1)), MONTHS[m.group(2)[:3].lower()]
    ey, em = int(end_date[:4]), int(end_date[5:7])
    year = ey if mon <= em else ey - 1
    return f"{year:04d}-{mon:02d}-{day:02d}"


def group_date(sub, end_date):
    """Last day a Championship League group was played, from its opening line ('Group 5 was played on 19 and 20
    January 2026'; the year may be missing, then it is the end date's year or the one before), or None."""
    m = re.search(r"(?:was|were) played on ([^.\n]*)", sub)
    if not m:
        return None
    days = re.findall(r"(\d{1,2})\s+([A-Z][a-z]+)(?:\s+(\d{4}))?", m.group(1))
    days = [d for d in days if d[1][:3].lower() in MONTHS]
    if not days:
        return None
    day, mon_name, year = days[-1]
    mon = MONTHS[mon_name[:3].lower()]
    ey, em = int(end_date[:4]), int(end_date[5:7])
    year = int(year) if year else (ey if mon <= em else ey - 1)
    return f"{year:04d}-{mon:02d}-{int(day):02d}"


def page_matches(text, end_date, played_by=None):
    """Matches on a page as (a, sa, b, sb, date, subsection, source); results under a 'Qualifying' level-2 heading get the
    qualifying date; in a Championship League, results under a group heading get that group's own date
    (group_date); subsection is the level-3 heading the result sits under ("" if none).
    Main-stage brackets carry no per-match dates, so those matches get the event's end date, or played_by (the
    page's revision date) if that is earlier: an event still running then has its results dated the day they were
    fetched instead of a future end date, which a forecast "as of" today would otherwise drop."""
    grouped = bool(re.search(r"Championship League", text[:3000]))  # league format: groups with their own dates
    text = clean(text)
    qdate = qualifying_date(text, end_date) or end_date
    main_date = min(end_date, played_by) if played_by else end_date
    out = []
    for sec in re.split(r"\n(?===[^=])", text):
        head = sec.split("\n", 1)[0]
        date = qdate if re.match(r"==[^=]*qualif", head, re.I) else main_date
        # A league group says when it was played ("Group 5 was played on 19 and 20 January 2026", in its level-2 or
        # level-3 section, and sometimes with a "====2 January====" heading per day): date its results then, not at
        # the end of a weeks-long event, so ratings "before" a match do not already contain later events.
        sec_gd = group_date(re.split(r"\n(?====[^=])", sec)[0], end_date) if grouped else None
        for sub in re.split(r"\n(?====[^=])", sec):  # level-3 parts, e.g. the groups of a Championship League
            label = sub.split("\n", 1)[0].strip("= ") if sub.startswith("===") else ""
            sub_gd = (group_date(sub, end_date) or sec_gd) if grouped else None
            for part in re.split(r"\n(?=====[^=])", sub):  # level-4 parts: one per day in some league pages
                gd = sub_gd
                if grouped and part.startswith("===="):
                    gd = group_date("was played on " + part.split("\n", 1)[0].strip("= "), end_date) or sub_gd
                d_ = (min(gd, played_by) if played_by else gd) if gd else date
                out += [(*m[:4], d_, label, m[4]) for m in section_matches(part)]
    return out


def canonical_ids(ids):
    """Merge variants like 'Ross Muir (snooker player)' / 'Ross Muir' and case variants; returns id -> name."""
    groups = defaultdict(Counter)
    for pid, n in ids.items():
        name = re.sub(r"\s*\((?:snooker player|snooker)\)$", "", pid)
        groups[name.casefold()][name] += n
    best = {k: c.most_common(1)[0][0] for k, c in groups.items()}
    return {pid: best[re.sub(r"\s*\((?:snooker player|snooker)\)$", "", pid).casefold()] for pid in ids}


def season_events(season_title, text):
    """Map event title -> (start date, end date, host country) from the WST calendar table."""
    y0 = int(season_title[:4])
    m = re.search(r"===\s*World Snooker Tour\s*===(.*?)\n===?[^=]", text, re.S)
    out = {}
    for row in (m.group(1) if m else "").split("\n|-"):
        cells = [c.strip() for c in re.split(r"\n\||\|\|", row) if c.strip()]
        link = next((link_parts(c) for c in cells if re.search(r"\[\[20\d\d ", c)), None)
        dates = [c for c in cells[:3] if re.fullmatch(r"\d{1,2}\s+[A-Za-z]{3}\w*", plain(c))]
        if not link or len(dates) < 2:
            continue
        ymd = []
        for cell in dates[:2]:
            day, mon = plain(cell).split()[:2]
            mon = MONTHS[mon[:3].lower()]
            ymd.append(f"{y0 if mon >= 6 else y0 + 1:04d}-{mon:02d}-{int(day):02d}")
        venue = next((c for c in cells if " in [[" in c or re.search(r",\s*[A-Z][a-z]", plain(c))), "")
        country = plain(venue).split(",")[-1].strip() if venue else ""
        out[link[0]] = (ymd[0], ymd[1], country)
    return out


def main():
    raw, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    seasons = json.loads((raw / "seasons.json").read_text())
    pages = json.loads((raw / "events.json").read_text())
    meta = {}
    for title, text in seasons.items():
        meta.update(season_events(title, text))

    rows, names, nats = [], defaultdict(Counter), defaultdict(Counter)
    for title, page in sorted(pages.items()):
        base = title.removesuffix(" qualifying")
        if base not in meta:
            print("no calendar entry for", title)
            continue
        _, end, country = meta[base]
        seen = set()
        # The same pair and score listed twice on a page is usually one match shown twice (bracket and final
        # section). In the Championship League the same pair can really meet again in a later group, often with
        # the same score, so there a repeat only counts as a duplicate inside one group (level-3 subsection).
        per_group = "Championship League" in base
        found = page_matches(page["content"], end, (page.get("timestamp") or "")[:10] or None)

        def same(a, sa, b, sb):  # pair and score; "X (snooker player)" and "X" are one player
            ka, kb = SUFFIX.sub("", a[0]), SUFFIX.sub("", b[0])
            return frozenset((ka, kb)), frozenset(((ka, sa), (kb, sb)))
        # A final table repeats the final when a bracket already has it. In a Championship League the same pair and
        # score can also come up in a league group, which is another match, so there only a bracket (the play-offs)
        # stands for the final; elsewhere a listed result does too.
        shown = {same(a, sa, b, sb) for a, sa, b, sb, _, _, src in found
                 if src == "bracket" or (src == "list" and not per_group)}
        for a, sa, b, sb, date, sub, src in found:
            if src == "final" and same(a, sa, b, sb) in shown:
                continue
            for pid, disp, nat in (a, b):
                names[disp][pid] += 1
                if nat:
                    nats[pid][nat.upper()] += 1
            if not isinstance(sa, int) or not isinstance(sb, int) or sa + sb == 0 or a[0] == b[0]:
                continue
            key = (*same(a, sa, b, sb), sub if per_group else None)
            if key in seen:
                continue
            seen.add(key)
            rows.append([date, base, country, a[0], b[0], sa, sb])

    # unlinked plain names (e.g. Championship League lists) -> linked article titles -> one id per person
    linked = {disp: c.most_common(1)[0][0] for disp, c in names.items()}
    counts = Counter()
    for r in rows:
        r[3], r[4] = linked.get(r[3], r[3]), linked.get(r[4], r[4])
        counts.update((r[3], r[4]))
    ids = canonical_ids(counts)
    nat_of = defaultdict(Counter)
    for pid, c in nats.items():
        nat_of[ids.get(linked.get(pid, pid), pid)].update(c)
    for r in rows:
        r[3], r[4] = ids[r[3]], ids[r[4]]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "matches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "event", "country", "p1", "p2", "f1", "f2", "nat1", "nat2"])
        for r in sorted(rows):
            n1 = nat_of[r[3]].most_common(1)[0][0] if nat_of[r[3]] else ""
            n2 = nat_of[r[4]].most_common(1)[0][0] if nat_of[r[4]] else ""
            w.writerow(r + [n1, n2])
    with open(out_dir / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "start_date", "end_date", "country"])
        for k, (s, e, c) in sorted(meta.items(), key=lambda kv: kv[1][1]):
            w.writerow([k, s, e, c])
    print(f"{len(rows)} matches, {len({r[3] for r in rows} | {r[4] for r in rows})} players")


if __name__ == "__main__":
    main()
