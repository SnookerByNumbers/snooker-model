"""Regression cases for the wikitext layouts parse_matches.py had to learn (one small snippet per past fix)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parse_matches as pm


def pairs(matches):
    """(player id, score, player id, score) of each parsed match, without display names and flags."""
    return [(m[0][0], m[1], m[2][0], m[3]) for m in matches]


BRACKET = """{{16TeamBracket
| RD1-seed01=1 | RD1-team01={{flagicon|ENG}} [[Judd Trump]] | RD1-score01=6
| RD1-seed02=16 | RD1-team02={{flagicon|CHN}} [[Si Jiahui]] | RD1-score02=4
| RD2-team01={{flagicon|ENG}} [[Judd Trump]] | RD2-score01=3
| RD2-team02={{flagicon|SCO}} [[John Higgins]] | RD2-score02=6
}}"""


class CleanTests(unittest.TestCase):
    def test_self_closing_ref_with_slash_in_name_keeps_the_bracket(self):
        # <ref name="OSullivan/Robertson" /> used to swallow everything up to the next </ref> (2017 Masters)
        text = 'Intro.<ref name="OSullivan/Robertson" />\n' + BRACKET + "\nLater.<ref>cite</ref>"
        self.assertEqual(pairs(pm.section_matches(pm.clean(text))),
                         [("Judd Trump", 6, "Si Jiahui", 4), ("Judd Trump", 3, "John Higgins", 6)])


class BracketTests(unittest.TestCase):
    def test_invoke_brackets(self):
        for opener in ("{{#invoke:RoundN|main", "{{#invoke: Team bracket | main", "{{#invoke:bracket|32TeamBracket-Info"):
            text = opener + "\n| RD1-team1=[[Mark Selby]] | RD1-score1=5\n| RD1-team2=[[Ali Carter]] | RD1-score2=2\n}}"
            self.assertEqual(pairs(pm.section_matches(text)), [("Mark Selby", 5, "Ali Carter", 2)], opener)

    def test_walkover_and_bye(self):
        text = ("{{8TeamBracket\n| RD1-team1=[[Mark Selby]] | RD1-score1=w/o\n| RD1-team2=[[Ali Carter]] | RD1-score2=w/d\n"
                "| RD1-team3=[[Kyren Wilson]] | RD1-score3=\n| RD1-team4=Bye | RD1-score4=\n}}")
        got = pm.section_matches(text)
        self.assertEqual(pairs(got), [("Mark Selby", "wo", "Ali Carter", "wo")])  # the bye is no match


class PlayerTests(unittest.TestCase):
    def test_nowrap_and_match_prefix(self):
        # Jiang Jun's league results were dropped while his name sat inside {{nowrap}}
        self.assertEqual(pm.parse_player("{{nowrap|{{flagicon|CHN}} [[Jiang Jun]]}}")[::2], ("Jiang Jun", "CHN"))
        self.assertEqual(pm.parse_player("Match 11: {{flagicon|POL}} Mateusz Baranowski")[0], "Mateusz Baranowski")

    def test_not_players(self):
        for s in ("[http://www.example.com Calendar]", "TBD", "", "{{Webarchive|url=x}}", "10–29"):
            self.assertIsNone(pm.parse_player(s), s)

    def test_disambiguation_suffix(self):
        self.assertEqual(pm.norm_pid("Ross Muir (snooker player)"), "Ross Muir")

    def test_spelling_variants_merge_to_the_most_used(self):
        merged = pm.canonical_ids({"Lü Chenwei": 3, "Lu Chenwei": 1, "Ng On-yee": 2, "Ng On Yee": 1, "Mark Selby": 9})
        self.assertEqual(merged["Lu Chenwei"], "Lü Chenwei")
        self.assertEqual(merged["Ng On Yee"], "Ng On-yee")
        self.assertEqual(len(set(merged.values())), 3)


class ListTests(unittest.TestCase):
    def test_result_line(self):
        self.assertEqual(pairs(pm.list_matches("* {{flagicon|ENG}} [[Judd Trump]] 4–2 {{flagicon|CHN}} [[Ding Junhui]]")),
                         [("Judd Trump", 4, "Ding Junhui", 2)])

    def test_frame_scores_after_arrow_are_not_part_of_the_name(self):
        line = "* [[Ronnie O'Sullivan]] 5–1 [[Neil Robertson]] → 77(57)–19, 113–0, 0–82, 70–1, 68–20, 90–5"
        self.assertEqual(pairs(pm.list_matches(line)), [("Ronnie O'Sullivan", 5, "Neil Robertson", 1)])

    def test_narrative_note_is_not_a_result(self):
        # counted as an extra match in the 2007 and 2010 Premier League
        line = ("* On 15 November [[Ronnie O'Sullivan]]'s 113 break in frame two of his 5–1 win over "
                "[[Neil Robertson]] was the 500th century break of his career.")
        self.assertEqual(pm.list_matches(line), [])

    def test_date_and_external_link_are_not_scores(self):
        self.assertEqual(pm.list_matches("* [http://www.example.com Calendar] {{Webarchive|url=x|date=2020-10-29}}"), [])
        self.assertEqual(pm.list_matches("* [[Mark Selby]] retrieved 2020-10-29 [[Ali Carter]]"), [])


class TableTests(unittest.TestCase):
    def test_three_cell_rows_also_with_header_score_cell(self):
        text = ("{|\n|-\n| '''[[Steve Davis]]'''\n| align=\"center\"| '''5'''–3\n| [[Jimmy White]]\n"
                "|-\n| [[Cliff Thorburn]]\n!scope=\"row\" | 10–7\n| [[Terry Griffiths]]\n|}")
        self.assertEqual(pairs(pm.table_matches(text)),
                         [("Steve Davis", 5, "Jimmy White", 3), ("Cliff Thorburn", 10, "Terry Griffiths", 7)])

    def test_final_table_is_read_once_and_marked_final(self):
        # brackets often stop at the semi-finals; the final sits in its own table (36 finals were missing)
        text = ("{| class=\"wikitable\"\n|-\n! colspan=\"3\" | '''Final:''' Best of 19 frames. Referee: X\n|-\n"
                "| width=\"230\" align=\"right\" | '''[[Wu Yize]]'''<br>{{CHN}}\n| align=\"center\" | '''10'''–6\n"
                "| width=\"230\" | [[John Higgins]]<br>{{SCO}}\n|-\n| colspan=\"3\" | 70–1, 0–85\n|}")
        got = pm.section_matches(text)
        self.assertEqual(pairs(got), [("Wu Yize", 10, "John Higgins", 6)])
        self.assertEqual(got[0][4], "final")


class DateTests(unittest.TestCase):
    def test_qualifying_date_wordings(self):
        end = "2024-02-04"
        for text, want in (
                ("Qualifiers took place from 18 to 22 December 2023 in Leicester.", "2023-12-22"),
                ("The event ran in Berlin.\n==Qualifying==\nThese matches were played between 18 and 22 December 2023.\n", "2023-12-22"),
                ("A two-round qualifying tournament was held from 18 to 22 December in Sheffield.", "2023-12-22")):
            self.assertEqual(pm.qualifying_date(text, end), want, text)

    def test_other_dates_near_qualifying_words_are_ignored(self):
        # the final's own date, or one more than 150 days back, is not the qualifying date
        self.assertIsNone(pm.qualifying_date("The qualifying winners met in the final on 4 February 2024.", "2024-02-04"))
        self.assertIsNone(pm.qualifying_date("Past qualifying formats changed on 1 March 2023.", "2024-02-04"))

    def test_league_group_date(self):
        self.assertEqual(pm.group_date("Group 5 was played on 19 and 20 January 2026.", "2026-02-05"), "2026-01-20")
        self.assertEqual(pm.group_date("Group 1 was played on 29 and 30 December.", "2026-02-05"), "2025-12-30")


class PageTests(unittest.TestCase):
    def test_stage_and_dates(self):
        text = ("The 2024 Test Open ended on 4 February.\n==Main draw==\n" + BRACKET +
                "\n==Qualifying==\nQualifying matches were played between 18 and 22 December 2023.\n"
                "* [[Mark Selby]] 5–2 [[Ali Carter]]\n")
        got = {(m[0][0], m[4], m[7]) for m in pm.page_matches(text, "2024-02-04")}
        self.assertEqual(got, {("Judd Trump", "2024-02-04", "main"), ("Mark Selby", "2023-12-22", "qualifying")})

    def test_running_event_is_dated_when_fetched(self):
        got = pm.page_matches("==Main draw==\n" + BRACKET, "2026-11-08", played_by="2026-11-03")
        self.assertEqual({m[4] for m in got}, {"2026-11-03"})

    def test_league_groups_keep_their_own_dates_and_labels(self):
        text = ("The 2026 Championship League is an invitational event.\n==Groups==\n"
                "===Group 1===\nGroup 1 was played on 2 and 3 January 2026.\n* [[Mark Selby]] 3–1 [[Ali Carter]]\n"
                "===Group 2===\nGroup 2 was played on 19 and 20 January 2026.\n* [[Mark Selby]] 3–1 [[Ali Carter]]\n")
        got = [(m[4], m[5]) for m in pm.page_matches(text, "2026-02-05")]
        self.assertEqual(got, [("2026-01-03", "Group 1"), ("2026-01-20", "Group 2")])

    def test_aggregate_points_event_is_skipped(self):
        text = "{{Infobox snooker tournament\n| score = 374–336 points\n}}\n==Main draw==\n" + BRACKET
        self.assertEqual(pm.page_matches(text, "1982-07-01"), [])

    def test_skipped_sections(self):
        text = "==Women's event==\n" + BRACKET + "\n==Main draw==\n* [[Mark Selby]] 5–2 [[Ali Carter]]\n"
        self.assertEqual(pairs(pm.page_matches(text, "2024-02-04")), [("Mark Selby", 5, "Ali Carter", 2)])


if __name__ == "__main__":
    unittest.main()
