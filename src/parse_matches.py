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
import unicodedata
from datetime import date
from collections import Counter, defaultdict
from pathlib import Path

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DROP_TEMPLATES = ("efn", "refn", "cite", "tooltip", "dagger", "abbr", "sfn", "cbignore", "small")
DASH = r"(?:–|—|-|\{\{\s*(?:ndash|nbnd|snd)\s*\}\}|&ndash;|&mdash;|&#821[12];|&#x201[34];)"
NOT_PLAYERS = {"bye", "tbd", "tba", "withdrawn", "w/d", "w/o", "walkover", ""}
SKIP = re.compile(r"Shoot.?Out|Mixed Doubles|Nations|Six-red|Seniors|Women|World Cup|Q School|Under-21|CVB Snooker Challenge"
                  r"|Pot Black|Team Classic|Doubles|Challenge Cup|Macau Masters|Ladbroke International"
                  r"|Pro Challenge Series – Event 2", re.I)  # single-frame, team, six-red and doubles events
SUFFIX = re.compile(r"\s*\((?:snooker(?:\s+player)?|snookern\s+player|pool\s+player|billiards\s+player|"
                    r"amateur\s+snooker\s+player|II)\)$", re.I)
ALIASES = {
    "Alfred Burden": "Alfie Burden", "Rory Thor": "Thor Chuan Leong", "Nutcharut Wongharuthai": "Mink Nutcharut",
    "Billy Castle": "Billy Joe Castle", "Mohammad Asif": "Muhammad Asif", "Joe Jogia": "Shailesh Jogia",
    "Pankaj Arjan Advani": "Pankaj Advani", "Mykhailo Larkov": "Michael Larkov",
    "Thanawat Thirapongpaiboon": "Thanawat Tirapongpaiboon", "T. Tirapongpaiboon": "Thanawat Tirapongpaiboon",
    "T.Tirapongpaiboon": "Thanawat Tirapongpaiboon", "M. Phetmalaikul": "Manasawin Phetmalaikul",
    "I Kachaiwong": "Issara Kachaiwong", "A Higginson": "Andrew Higginson", "A. Higginson": "Andrew Higginson",
    "Ishpreet Chadha": "Ishpreet Singh Chadha", "Jon Birch": "Jonathan Birch", "Jonathan Fulcher": "Jonni Fulcher",
    "Bob Marshall": "Robert Marshall", "Bill Kelly": "Billy Kelly", "Anita Rizzuti": "Anita Maflin",
    "Chuchart Trairatanapradit": "Tai Pichit", "David Finbow": "Dave Finbow",
    "Samuel Thistlewhite": "Sam Thistlewhite", "Anthony Bolsover": "Antony Bolsover", "Frankie Chan": "Franky Chan",
    "Sammy Chong": "Sam Chong", "Phil Seaton": "Philip Seaton", "Phillip Seaton": "Philip Seaton",
    "Phil Tarrant": "Philip Tarrant", "Phillip Tarrant": "Philip Tarrant", "Phil Williams": "Philip Williams",
    "Sidney Wilson": "Sydney Wilson", "George Pragnall": "George Pragnell", "Sergey Isaenko": "Sergiy Isayenko",
    "Himanshu Dinesh Jain": "Himanshu Jain", "Ian Barry Stark": "Ian Stark",
    "Ivelin Boyanov Bozhanov": "Ivelin Bozhanov", "Ivelin Boyanov": "Ivelin Bozhanov",
    "Mohammed Joker": "Mohammed Al Joker", "Tang": "Tang Jun",
    # Wikipedia typos
    "Lei be Peifan": "Lei Peifan", "Davild Gilbert": "David Gilbert", "Dave Gilbert": "David Gilbert",
    "Anthofny Hamilton": "Anthony Hamilton", "Matthews Stevens": "Matthew Stevens", "Ben Woolaston": "Ben Woollaston",
    "Mike Dun": "Mike Dunn", "Gerrard Greene": "Gerard Greene", "Micheal Judge": "Michael Judge",
    "Jame McGouran": "James McGouran", "Adamn King": "Adam King", "Adam Egde": "Adam Edge",
    "Williams Lemons": "William Lemons", "William Lemmons": "William Lemons", "Khishan Hirani": "Kishan Hirani",
    "Amman Iqbal": "Amaan Iqbal", "Aymn Alamri": "Ayman Alamri", "Abdulraouf Sayegh": "Abdulraouf Saigh",
    "Ali Alobaidli": "Ali Al Obaidly", "Johnathan Bagley": "Jonathan Bagley", "Aaran Hetherington": "Aran Hetherington",
    "Hans Blankaert": "Hans Blanckaert", "Daan Leyssen": "Daan Leysen", "Felix Freede": "Felix Frede",
    "Markus Fisher": "Markus Fischer", "Michele Bataglia": "Michele Battaglia",
    "Olivier Vandedoheede": "Olivier Vandenboheede", "Olivier Vandenbohede": "Olivier Vandenboheede",
    "Ralf Hemmerlingx": "Ralf Hemmerling", "Robert Thickner": "Robert Tickner", "Ronny Buckholz": "Ronny Buchholz",
    "Sascha Brauer": "Sascha Breuer", "Stephen Ellise": "Stephen Ellis", "Thomas Whealthy": "Thomas Wealthy",
    "Tim Van Goubergen": "Tim Van Gouberghen", "Tom Vanwelssenaers": "Tom Vanwelsenaers",
    "Vladimir Potaznyk": "Vladimir Potazsnyk", "Wan Chooi Tang": "Wan Chooi Tan",
    "Wiktor Doberschotz": "Wiktor Doberschütz", "Mario Wehrman": "Mario Wehrmann", "John Dunnning": "John Dunning",
    "Eddie McLauglin": "Eddie McLaughlin", "Davis Gray": "David Gray", "Jamie O'Neil": "Jamie O'Neill",
    "Mohammed Yousuf": "Muhammad Yousaf", "Alastair Fleming": "Alistair Fleming", "Gary Baldrey": "Garry Baldrey",
    "Gay Burns": "Gaye Burns", "Lyu Chenwei": "Lu Chenwei", "Matthais Porn": "Matthias Porn",
    "Garry Steele": "Gary Steele", "Anthony Parsons": "Antony Parsons",
}


def norm_pid(pid):
    if re.search(r"born\s+1961", pid, re.I):
        return "Dave Gilbert (born 1961)"
    p = SUFFIX.sub("", pid).strip()
    p = re.sub(r"^([A-Z][a-z]+)\s+[A-Z]\.?\s+([A-Z][a-z]+)$", r"\1 \2", p)
    return ALIASES.get(p, p)


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
    text = re.sub(r"<ref\b[^>]*?/>", "", text)  # self-closing, also with a "/" in the name ("OSullivan/Robertson")
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
    target = re.sub(r"^:[a-z]+:\s*", "", m.group(1).strip(), flags=re.I)
    target = target[0].upper() + target[1:] if target else target
    return target, (m.group(2) or re.sub(r"\s*\(.*?\)$", "", target)).strip()


def plain(s):
    s = re.sub(r"\{\{[^{}]*\}\}", " ", s)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("'''", "").replace("''", "")
    s = re.sub(r"\(\s*(?:\d+|a|wc|WC|q)\s*\)", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .,:;*†‡")


def unwrap(s, names=("nowrap", "nobr", "not a typo")):
    """Replace {{nowrap|X}} by X (older pages wrap plain names: "{{nowrap|{{flagicon|POL}} Mateusz Baranowski}}")."""
    pat = re.compile(r"\{\{\s*(?:" + "|".join(names) + r")\s*\|", re.I)
    while (m := pat.search(s)):
        end = template_end(s, m.start())
        s = s[:m.start()] + s[m.end():end - 2] + s[end:]
    return s


def parse_player(s):
    """Return (id, display, nationality) for a player cell, or None."""
    s = unwrap(s).replace("'''", "").replace("''", "")
    s = re.sub(r"^\s*(?:Match\s+\d+|(?:Semi-)?Final\w*|Third\s+place\s+match|PQ\d*)\s*:\s*", "", s, flags=re.I)
    m = re.search(r"\{\{\s*flagathlete\s*\|", s, re.I)
    if m:
        inner = split_top(s[m.end():template_end(s, m.start()) - 2])
        name, nat = inner[0], (inner[1].strip() if len(inner) > 1 else "")
        lp = link_parts(name)
        if lp:
            if lp[1].lower() in NOT_PLAYERS or lp[0].lower().startswith("bye"):
                return None
            return lp[0], lp[1], nat
        name = plain(name).split(",")[0].strip()
        return (name, name, nat) if name.lower() not in NOT_PLAYERS else None
    nat = ""
    m = re.search(r"\{\{\s*flagicon\s*\|\s*([A-Za-z]+)", s, re.I) or re.search(r"\{\{\s*([A-Z]{3})\s*\}\}", s)
    if m:
        nat = m.group(1)
    lp = link_parts(s)
    if lp:
        if lp[1].lower() in NOT_PLAYERS or lp[0].lower().startswith("bye"):
            return None
        return lp[0], lp[1], nat
    name = plain(s).split(",")[0].strip()
    if not name or name.lower() in NOT_PLAYERS or re.fullmatch(r"[\d\s/()a-z.–-]*", name):
        return None
    if re.search(r"https?:|www\.", s) or re.search(r"[{}\[\]|=]", name) or not re.search(r"[A-Za-z]", name):
        return None  # a web link or template remnant, not a name (external-link lists at the foot of a page)
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


NARRATIVE = re.compile(r"\b(?:win|wins|won|beat|beats|defeated|losing|lost|led|came\s+from|runner-up|"
                       r"shoot-?out|break|century|centuries|frames?|was|were|his|her|their|career)\b", re.I)


def list_matches(text):
    out = []
    for line in text.splitlines():
        if not line.lstrip().startswith(("*", "#")):
            continue
        line = line.lstrip("*# ").replace("'''", "").replace("''", "")
        if line.startswith("[http"):  # external links ("* [http://... Calendar] {{Webarchive|...|date=2020-10-29}}")
            continue
        # "A 5–1 B → 77(57)–19, ...": the frame scores are not part of B (also when "→" is omitted before frame scores)
        line = re.split(r"→|\s+\d{1,3}\s*–\s*(?:\d{2,3}|\(\d{3}\))\s*(?:\(\d+\)\s*)?,", line)[0]
        # a note, not a result: "On 15 November [[Ronnie O'Sullivan]]'s 113 break in frame two of his 5–1 win over
        # [[Neil Robertson]] was the 500th century break of his career." (links left out; "(one-frame shoot-out)" kept)
        if NARRATIVE.search(re.sub(r"\[\[[^\]]*\]\]", "", line)):
            continue
        # a score is not part of a date: "2020-10-29" holds "10-29"
        scores = list(re.finditer(r"(?<![\d,£\-–])(\d{1,2})\s*" + DASH + r"\s*(\d{1,2})(?![\d,\-–])", line))
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
            cells = [re.sub(r"</?(?:b|br\s*/?)>", "", x, flags=re.I) for x in cells]
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
    # bracket templates: {{16TeamBracket|...}}, {{#invoke:RoundN|...}}, and {{#invoke: Team bracket | main |...}}
    # (2016-2019 World Championship qualifying)
    # also {{#invoke:bracket|32TeamBracket-Info|...}} (1990s pages) and the old {{Round16|...}} family (1970s)
    for m in re.finditer(r"\{\{\s*(?:[^|{}]*Bracket[^|{}]*|Round\d+[^|{}]*|#invoke:\s*(?:RoundN|[Tt]eam [Bb]racket|[Bb]racket)[^|{}]*)\|",
                         text):
        if m.start() < pos:
            continue
        end = template_end(text, m.start())
        rest.append(text[pos:m.start()])
        out += bracket_matches(text[m.start() + 2:end - 2])
        pos = end
    rest.append(text[pos:])
    rest = "".join(rest)
    # source of each result: "bracket", "list", or "final" (a final read from its own table, which only fills in a
    # final the page shows nowhere else). Three-cell result tables count as "list"; a final table is one too, so
    # table_matches leaves out what final_tables already read.
    finals = final_tables(rest)
    tables = [m for m in table_matches(rest) if m not in finals]
    return [(*m, "bracket") for m in out] + [(*m, "list") for m in list_matches(rest) + tables] + \
        [(*m, "final") for m in finals]


def table_matches(text):
    """Results in wikitable rows of three cells, player | score | player. Older pages list qualifying rounds this
    way, e.g. a row of A in bold, then align="center"| 5–3 (winner's score in bold), then B."""
    out = []
    for tbl in re.finditer(r"^\{\|.*?^\|\}", text, re.M | re.S):
        for row in re.split(r"\n\|-[^\n]*", tbl.group()):
            # a row's cells, also header-style ones: 1980s pages put the score in a "!scope="row" | 10–7" cell
            cells = [cell(x) for line in row.strip("\n").split("\n")
                     if line[:1] in ("|", "!") and not line.startswith(("|-", "|}", "|+", "{|"))
                     for x in re.split(r"\|\||!!", line[1:])]
            if len(cells) != 3:
                continue
            cells = [re.sub(r"</?(?:b|br\s*/?)>", "", x, flags=re.I) for x in cells]
            sc = re.fullmatch(r"\s*(\d{1,2})\s*" + DASH + r"\s*(\d{1,2})\s*", cells[1].replace("'''", ""))
            if not sc:
                continue
            a, b = parse_player(cells[0]), parse_player(cells[2])
            if a and b:
                out.append((a, int(sc.group(1)), b, int(sc.group(2))))
    return out


MONTH_NAME = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?" \
             r"|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
# "28 March", "22 and 23 November", "9–11 February 2016", "from 22 to 27 November", "31 May and 2 June 2017"
DATE_EXPR = re.compile(r"\b(\d{1,2})(?:\s+(" + MONTH_NAME + r"))?(?:\s*(?:–|—|-|to|and|&)\s*(\d{1,2}))?\s+("
                       + MONTH_NAME + r")\b(?:,?\s+(\d{4}))?")


def _date_in(s, end_date):
    """End of the first date expression in s ('9–11 February 2016' -> the 11th), as YYYY-MM-DD, if it falls 3 to
    150 days before end_date (the qualifiers come before the main stage; anything else is some other date)."""
    m = DATE_EXPR.search(s.replace("&nbsp;", " ").replace("\xa0", " "))  # "between 5&nbsp;and 14&nbsp;April"
    if not m:
        return None
    day, mon = int(m.group(3) or m.group(1)), MONTHS[m.group(4)[:3].lower()]
    ey, em = int(end_date[:4]), int(end_date[5:7])
    year = int(m.group(5)) if m.group(5) else (ey if mon <= em else ey - 1)
    try:
        d = date(year, mon, day)
    except ValueError:
        return None
    end = date.fromisoformat(end_date)
    return d.isoformat() if (end - d).days in range(3, 151) else None  # the final's own date is not qualifying


def qualifying_date(text, end_date):
    """Last day of qualifying, or None. First the lead ('Qualifiers took place from 17 to 19 September'); then,
    since pages word it many ways ('Qualifying for the 2019 ... took place from 10 to 17 April', 'These matches
    were played between 31 May and 2 June 2017', 'a four-round qualifying tournament from 22 to 27 November'),
    the first date in a qualifying section, then in any sentence that mentions qualifying. Without this the
    qualifiers got the main stage's date, so a backtest took them for main-stage matches (62 events, 2015-2026)."""
    m = re.search(r"Qualif\w*\s+(?:took|take|were|was|are|is|will)[^.]*?(\d{1,2})\s+([A-Z][a-z]+)", text)
    if m and m.group(2)[:3].lower() in MONTHS:
        day, mon = int(m.group(1)), MONTHS[m.group(2)[:3].lower()]
        ey, em = int(end_date[:4]), int(end_date[5:7])
        year = ey if mon <= em else ey - 1
        return f"{year:04d}-{mon:02d}-{day:02d}"
    for sec in re.split(r"\n(?===)", text):
        head = sec.split("\n", 1)[0]
        if re.match(r"==+[^=]*qualif", head, re.I) and not re.search(r"centur", head, re.I):
            found = _date_in(sec.split("\n", 1)[-1], end_date)
            if found:
                return found
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if re.search(r"qualif", sentence, re.I):
            found = _date_in(sentence, end_date)
            if found:
                return found
    return None


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


SKIP_SECTION = re.compile(r"\b(?:women|ladies|billiards|associated tournaments|tournament summary|team event|six-red)\b", re.I)


def page_matches(text, end_date, played_by=None):
    """Matches on a page as (a, sa, b, sb, date, subsection, source, stage); results under a 'Qualifying' level-2 heading
    get the qualifying date and stage "qualifying" (else "main"); in a Championship League, results under a group heading get that group's own date
    (group_date); subsection is the level-3 heading the result sits under ("" if none).
    Main-stage brackets carry no per-match dates, so those matches get the event's end date, or played_by (the
    page's revision date) if that is earlier: an event still running then has its results dated the day they were
    fetched instead of a future end date, which a forecast "as of" today would otherwise drop."""
    if re.search(r"\|\s*score\s*=\s*[^\n|]*(?:points|aggregate)", text[:3000], re.I):
        return []  # aggregate-points tournament (e.g. 1982 Australian Masters), not frame scores
    grouped = bool(re.search(r"Championship League", text[:3000]))  # league format: groups with their own dates
    text = clean(text)
    qdate = qualifying_date(text, end_date) or end_date
    main_date = min(end_date, played_by) if played_by else end_date
    out = []
    for sec in re.split(r"\n(?===[^=])", text):
        head = sec.split("\n", 1)[0]
        if SKIP_SECTION.search(head):
            continue
        qualifying = bool(re.match(r"==[^=]*qualif", head, re.I))
        date = qdate if qualifying else main_date
        # A league group says when it was played ("Group 5 was played on 19 and 20 January 2026", in its level-2 or
        # level-3 section, and sometimes with a "====2 January====" heading per day): date its results then, not at
        # the end of a weeks-long event, so ratings "before" a match do not already contain later events.
        sec_gd = group_date(re.split(r"\n(?====[^=])", sec)[0], end_date) if grouped else None
        for sub in re.split(r"\n(?====[^=])", sec):  # level-3 parts, e.g. the groups of a Championship League
            label = sub.split("\n", 1)[0].strip("= ") if sub.startswith("===") else ""
            if SKIP_SECTION.search(label) or re.search(r"amateur", label, re.I) and not qualifying:
                continue
            sub_gd = (group_date(sub, end_date) or sec_gd) if grouped else None
            for part in re.split(r"\n(?=====[^=])", sub):  # level-4 parts: one per day in some league pages
                gd = sub_gd
                if grouped and part.startswith("===="):
                    gd = group_date("was played on " + part.split("\n", 1)[0].strip("= "), end_date) or sub_gd
                d_ = (min(gd, played_by) if played_by else gd) if gd else date
                out += [(*m[:4], d_, label, m[4], "qualifying" if qualifying else "main")
                        for m in section_matches(part)]
    return out


def name_key(name):
    """'Lü Chenwei', 'Lu Chenwei'; 'Yu De Lu', 'Yu Delu'; 'Ng On-yee', 'Ng On Yee' -> one key: accents, case,
    spaces and punctuation do not tell two players apart."""
    s = name.lower()
    for a, b in (("ł", "l"), ("ø", "o"), ("ß", "ss"), ("đ", "d"), ("æ", "ae"), ("œ", "oe")):
        s = s.replace(a, b)
    folded = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    folded = re.sub(r"[^a-z]", "", folded)
    for a, b in (("ae", "a"), ("oe", "o"), ("ue", "u")):
        folded = folded.replace(a, b)
    return folded


def canonical_ids(ids):
    """Merge variants like 'Ross Muir (snooker player)' / 'Ross Muir', and spelling variants that only differ in
    accents, case, spaces or hyphens; the most used spelling wins. Returns id -> name."""
    groups = defaultdict(Counter)
    for pid, n in ids.items():
        cp = norm_pid(pid)
        groups[name_key(cp)][cp] += n
    best = {k: c.most_common(1)[0][0] for k, c in groups.items()}
    return {pid: best[name_key(norm_pid(pid))] for pid in ids}


def season_events(season_title, text):
    """Map event title -> (start date, end date, host country) from the WST calendar table."""
    y0 = int(season_title[:4])
    m = re.search(r"===\s*World Snooker Tour\s*===(.*?)\n===?[^=]", text, re.S)
    out = {}
    for row in (m.group(1) if m else "").split("\n|-"):
        cells = [c.strip() for c in re.split(r"\n\||\|\|", row) if c.strip()]
        link = next((link_parts(c) for c in cells if re.search(r"\[\[20\d\d ", c)), None)
        dates = [c for c in cells[:3] if re.fullmatch(r"\d{1,2}\s+[A-Za-z]{3}\w*", plain(c))]
        if len(dates) < 2:
            continue
        if not link:  # older seasons: titles without a leading year ("European Tour 2015/2016 – Event 1"); take the
            # tournament cell, the one right after the start and end dates. Only that cell: when the tournament has
            # no article, the next link is the venue ("Antalya"), not an event.
            after = cells[cells.index(dates[1]) + 1:]
            link = link_parts(after[0]) if after else None
        if not link:
            continue
        ymd = []
        for cell in dates[:2]:
            day, mon = plain(cell).split()[:2]
            mon = MONTHS[mon[:3].lower()]
            ymd.append(f"{y0 if mon >= 6 else y0 + 1:04d}-{mon:02d}-{int(day):02d}")
        # The season rule (June-December = first year) fails for events moved past the season's end: the 2020 World
        # Championship sits in the 2019-20 calendar but was played in July-August 2020. When the title's own year
        # disagrees with both dates, shift them to it.
        ty = re.match(r"(20\d\d) ", link[0])
        if ty and all(int(x[:4]) != int(ty.group(1)) for x in ymd):
            shift = int(ty.group(1)) - int(ymd[1][:4])
            ymd = [f"{int(x[:4]) + shift:04d}{x[4:]}" for x in ymd]
        venue = next((c for c in cells if " in [[" in c or re.search(r",\s*[A-Z][a-z]", plain(c))), "")
        country = plain(venue).split(",")[-1].strip() if venue else ""
        out[link[0]] = (ymd[0], ymd[1], country)
    # Older calendars (before about 2010) are read by their header instead: no "World Snooker Tour" subsection
    # before 1996-97 (the table sits right under "Calendar"), dates as "{{nowrap|9 Sep}}" or "08-21" (month-day,
    # "08-??" when the day is unknown), a flag column between the dates and the tournament, and titles starting
    # with 19xx. Events found above keep their entry; these only add what the rows above missed.
    for k, v in calendar_by_header(y0, text).items():
        out.setdefault(k, v)
    return out


FLAG_COUNTRY = {"CHN": "China", "ENG": "England", "SCO": "Scotland", "WAL": "Wales", "NIR": "Northern Ireland",
                "IRL": "Ireland", "THA": "Thailand", "HKG": "Hong Kong", "GER": "Germany", "MLT": "Malta",
                "CAN": "Canada", "AUS": "Australia", "BEL": "Belgium", "IND": "India", "LAT": "Latvia",
                "GIB": "Gibraltar", "POL": "Poland", "BUL": "Bulgaria", "TUR": "Turkey", "KSA": "Saudi Arabia",
                "UAE": "United Arab Emirates", "MAR": "Morocco", "SGP": "Singapore", "MAC": "Macau",
                "NZL": "New Zealand", "RSA": "South Africa", "AUT": "Austria", "SUI": "Switzerland", "FIN": "Finland",
                "ISR": "Israel", "BRA": "Brazil", "MAS": "Malaysia", "PAK": "Pakistan", "ESP": "Spain",
                "NED": "Netherlands", "FRA": "France", "CYP": "Cyprus", "QAT": "Qatar", "BHR": "Bahrain",
                "EGY": "Egypt", "UKR": "Ukraine", "RUS": "Russia", "SWE": "Sweden", "NOR": "Norway", "DEN": "Denmark",
                "USA": "United States", "JPN": "Japan", "KOR": "South Korea", "TPE": "Taiwan"}


def _calendar_date(cell, y0):
    """(month, day) from a calendar date cell: "9 Sep", "{{nowrap|9 Sep}}", "08-21", "08–21", "08-??"; or None."""
    s = plain(unwrap(cell))
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]{3})\w*", s)
    if m and m.group(2)[:3].lower() in MONTHS:
        return MONTHS[m.group(2)[:3].lower()], int(m.group(1))
    m = re.fullmatch(r"(\d{1,2})\s*[-–]\s*(\d{1,2}|\?\?)", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(1)), (int(m.group(2)) if m.group(2).isdigit() else None)
    return None


def calendar_by_header(y0, text):
    """Event title -> (start, end, country) from calendar tables, locating the date and tournament columns by
    their header ("Start"/"Finish" or a two-column "Date"; "Tournament name"). Uses the "World Snooker Tour"
    subsection when there is one, otherwise the "Calendar" section. Unknown days ("08-??") become the 1st (start)
    and the 28th (end); years follow the season rule, then the title's own year (see season_events)."""
    m = re.search(r"===\s*World Snooker Tour\s*===(.*?)(?:\n===?[^=]|\Z)", text, re.S) or \
        re.search(r"==\s*Calendar\s*==(.*?)(?:\n===?[^=]|\Z)", text, re.S)
    out = {}
    if not m:
        return out
    for tbl in re.finditer(r"\{\|.*?\n\|\}", m.group(1), re.S):
        parts = tbl.group().split("\n|-")
        head = []
        for line in parts[0].split("\n") + (parts[1].split("\n") if len(parts) > 1 and "\n!" in parts[1] else []):
            if line.startswith("!"):
                for h in re.split(r"!!", line[1:]):
                    span = re.search(r"colspan\s*=\s*\"?(\d+)", h.split("|")[0] if "|" in h else "")
                    label = plain(h.split("|")[-1]).lower()
                    head += [label] * (int(span.group(1)) if span else 1)
        dcols = [i for i, h in enumerate(head) if re.match(r"(date|start|finish|end)", h)]
        tcols = [i for i, h in enumerate(head) if "tournament" in h or h == "event"]
        ccols = [i for i, h in enumerate(head) if h in ("country", "venue", "city", "location")]
        if len(dcols) < 2 or not tcols:
            continue
        for row in parts[1:]:
            cells = [cell(c).strip() for line in row.split("\n") if line.startswith("|") and not line.startswith("|}")
                     for c in line[1:].split("||")]
            if len(cells) != len(head):  # a colspan/rowspan row ("Cancelled", a shared date) does not line up
                continue
            d0, d1 = _calendar_date(cells[dcols[0]], y0), _calendar_date(cells[dcols[1]], y0)
            link = link_parts(unwrap(cells[tcols[0]]))
            if not (d0 and d1 and link) or re.search(r"\[\[[^\]|]*#", cells[tcols[0]]):
                continue  # "[[2006 Masters (snooker)#Qualifying|...]]" is a qualifying event, not the Masters itself
            ev_title = link[0]
            m_ev = re.search(r"–\s*Event\s+\d+", plain(cells[tcols[0]]))
            if m_ev and "Event" not in ev_title:
                ev_title = f"{ev_title} {m_ev.group(0)}"
            ymd = [f"{y0 if mo >= 6 else y0 + 1:04d}-{mo:02d}-{(dd or (1 if k == 0 else 28)):02d}"
                   for k, (mo, dd) in enumerate((d0, d1))]
            ty = re.match(r"((?:19|20)\d\d) ", ev_title)
            if ty and all(int(x[:4]) != int(ty.group(1)) for x in ymd):
                shift = int(ty.group(1)) - int(ymd[1][:4])
                ymd = [f"{int(x[:4]) + shift:04d}{x[4:]}" for x in ymd]
            if ymd[0] > ymd[1]:
                if ty and int(ty.group(1)) == y0 + 1 and int(ymd[0][:4]) == y0 + 1:
                    ymd[1] = f"{y0 + 1:04d}{ymd[1][4:]}"
                else:
                    ymd[0] = f"{int(ymd[0][:4]) - 1:04d}{ymd[0][4:]}"
            country = ""
            for i in ccols:
                f = re.search(r"\{\{\s*flag(?:icon|country)?\s*\|\s*([A-Z]{3})", cells[i])
                if f:
                    country = FLAG_COUNTRY.get(f.group(1), f.group(1))
                    break
            if not country:
                venue = " ".join(plain(cells[i]) for i in ccols)
                country = venue.split(",")[-1].strip() if "," in venue else ""
            out.setdefault(ev_title, (ymd[0], ymd[1], country))
    return out


def calendar(raw):
    """Event title -> (start, end, country) over all season calendars in RAW_DIR/seasons.json, minus SKIP."""
    meta = {}
    for title, text in json.loads((Path(raw) / "seasons.json").read_text()).items():
        for k, v in season_events(title, text).items():
            if not SKIP.search(k):
                meta[k] = v
    return meta


def expand_pages(pages):
    """Split multi-event pages ('Pro Challenge Series' -> 'Pro Challenge Series – Event 1/2/3/5') into one entry per event."""
    out = dict(pages)
    if "Pro Challenge Series" in out:
        pcs = out.pop("Pro Challenge Series")
        parts = re.split(r"^==\s*(Event \d+)\s*==\s*$", pcs["content"], flags=re.M)
        for ev_num, body in zip(parts[1::2], parts[2::2]):
            sub_title = f"Pro Challenge Series – {ev_num}"
            if not SKIP.search(sub_title):
                promoted = re.sub(r"^=(==+[^\n]*?)=", r"\1", body, flags=re.M)
                out[sub_title] = {"content": promoted, "timestamp": pcs.get("timestamp"), "requested": sub_title}
    return out


def event_name(title, page, meta):
    """The event a page is filed under in matches.csv (and centuries.csv): the calendar's title, with a separate
    "X qualifying" page under X, and a redirect under the name the calendar linked ("2003 British Open (snooker)"
    for the page "2003 British Open"). None if the calendar does not list it."""
    base = title.removesuffix(" qualifying")
    if base not in meta and page.get("requested", "").removesuffix(" qualifying") in meta:
        base = page["requested"].removesuffix(" qualifying")  # the calendar linked a redirect to this page
    return base if base in meta else None


def main():
    raw, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    pages = expand_pages(json.loads((raw / "events.json").read_text()))
    meta = calendar(raw)

    rows, names, nats = [], defaultdict(Counter), defaultdict(Counter)
    for title, page in sorted(pages.items()):
        if SKIP.search(title):
            continue
        base = event_name(title, page, meta)
        if base is None:
            print("no calendar entry for", title)
            continue
        _, end, country = meta[base]
        seen = set()
        # The same pair and score listed twice on a page is usually one match shown twice (bracket and final
        # section). In the Championship League the same pair can really meet again in a later group, often with
        # the same score, so there a repeat only counts as a duplicate inside one group (level-3 subsection).
        per_group = "Championship League" in base
        found = page_matches(page["content"], end, (page.get("timestamp") or "")[:10] or None)
        ev_full = {norm_pid(p[0]) for a, sa, b, sb, *_ in found if isinstance(sa, int) and isinstance(sb, int)
                   for p in (a, b) if " " in p[1].strip()}

        def resolve_short(p):
            pid, disp, nat = p
            if " " not in disp.strip():
                cands = [fp for fp in ev_full if disp.lower() in [t.lower() for t in fp.split()]]
                if len(cands) == 1:
                    return cands[0], cands[0], nat
            return pid, disp, nat

        found = [(resolve_short(a), sa, resolve_short(b), sb, date, sub, src, stage)
                 for a, sa, b, sb, date, sub, src, stage in found]

        def same(a, sa, b, sb):  # pair and score; "X (snooker player)" and "X" are one player
            ka, kb = norm_pid(a[0]), norm_pid(b[0])
            return frozenset((ka, kb)), frozenset(((ka, sa), (kb, sb)))
        # A final table repeats the final when a bracket already has it. In a Championship League the same pair and
        # score can also come up in a league group, which is another match, so there only a bracket (the play-offs)
        # stands for the final; elsewhere a listed result does too.
        shown = {same(a, sa, b, sb) for a, sa, b, sb, _, _, src, _ in found
                 if src == "bracket" or (src == "list" and not per_group)}
        for a, sa, b, sb, date, sub, src, stage in found:
            if title.endswith(" qualifying"):  # a separate qualifying page, filed under its main event
                stage = "qualifying"
            if src == "final" and same(a, sa, b, sb) in shown:
                continue
            for pid, disp, nat in (a, b):
                names[disp][pid] += 1
                if nat:
                    nats[pid][nat.upper()] += 1
            if not isinstance(sa, int) or not isinstance(sb, int) or sa + sb == 0 or norm_pid(a[0]) == norm_pid(b[0]):
                continue
            key = (*same(a, sa, b, sb), sub if per_group else None)
            if key in seen:
                continue
            seen.add(key)
            rows.append([date, base, country, a[0], b[0], sa, sb, stage])

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
        w.writerow(["date", "event", "country", "p1", "p2", "f1", "f2", "nat1", "nat2", "stage"])
        for r in sorted(rows):
            n1 = nat_of[r[3]].most_common(1)[0][0] if nat_of[r[3]] else ""
            n2 = nat_of[r[4]].most_common(1)[0][0] if nat_of[r[4]] else ""
            w.writerow(r[:7] + [n1, n2, r[7]])
    with open(out_dir / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "start_date", "end_date", "country"])
        for k, (s, e, c) in sorted(meta.items(), key=lambda kv: kv[1][1]):
            w.writerow([k, s, e, c])
    print(f"{len(rows)} matches, {len({r[3] for r in rows} | {r[4] for r in rows})} players")


if __name__ == "__main__":
    main()
