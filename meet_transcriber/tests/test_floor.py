import os
import sys
import unittest
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from caption_feed import ROLE_FLOOR, ROLE_SIDE, CaptionLineFeed  # noqa: E402

Gloss = namedtuple("Gloss", "gloss start end")


def motion(frames):
    frame = [0.0] * 258
    frame[0] = 0.1
    return [list(frame) for _ in range(frames)]


class FloorTests(unittest.TestCase):
    def make(self, seconds=1, signs=True):
        self.sent, self.t = [], 0.0
        if not signs:
            return CaptionLineFeed(self.sent.append, now=lambda: self.t)
        feed = CaptionLineFeed(self.sent.append, translate=lambda text: [Gloss("hi", 0, 2)],
                               load_motion=lambda g: motion(round(seconds * 30)),
                               now=lambda: self.t)
        return feed

    def types(self, kind):
        return [m for m in self.sent if m["type"] == kind]

    def floors(self):
        return [m["speaker"] for m in self.types("floor")]

    def test_first_speaker_takes_the_floor_before_their_line(self):
        feed = self.make()
        feed.interim(1, "Ann", "hi")
        self.assertEqual([m["type"] for m in self.sent], ["floor", "line"])
        self.assertEqual(self.sent[0], {"v": 1, "type": "floor", "speaker": "Ann"})
        self.assertEqual(self.sent[1]["role"], ROLE_FLOOR)

    def test_second_speaker_while_floor_is_held_is_a_side_comment_and_not_signed(self):
        feed = self.make()
        feed.interim(1, "Ann", "hi")
        feed.final(2, "Bob", "hi")
        bob = [m for m in self.types("line") if m["speaker"] == "Bob"]
        self.assertEqual([m["role"] for m in bob], [ROLE_SIDE])
        self.assertEqual(self.floors(), ["Ann"])
        self.assertEqual(self.types("sign"), [])

    def test_role_is_fixed_when_the_line_is_created(self):
        feed = self.make()
        feed.interim(2, "Bob", "hi")           # Bob holds the floor
        feed.interim(1, "Ann", "hey")          # Ann's line is a side comment
        feed.final(2, "Bob", "hi.")
        self.t = 100.0
        feed.tick()                            # floor released
        feed.final(1, "Ann", "hey.")           # still a side comment, never signed
        ann = [m for m in self.types("line") if m["speaker"] == "Ann"]
        self.assertEqual([m["role"] for m in ann], [ROLE_SIDE, ROLE_SIDE])
        self.assertEqual([m["line"] for m in self.types("sign")], [1])

    def test_an_open_line_keeps_its_role_after_falling_out_of_the_snapshot(self):
        feed = self.make()
        feed.interim(1, "Ann", "hi")
        feed.interim(2, "Bob", "hmm")          # side comment, stays open
        for i in range(20):
            feed.final(100 + i, "Ann", "x")
        feed.interim(2, "Bob", "hmm hmm")
        self.assertEqual([m["role"] for m in self.types("line") if m["id"] == 2], [ROLE_SIDE, ROLE_SIDE])
        self.assertEqual(self.floors(), ["Ann"])

    def test_holder_final_lines_are_signed(self):
        feed = self.make()
        feed.final(1, "Ann", "hi")
        self.assertEqual([m["line"] for m in self.types("sign")], [1])

    def test_floor_is_kept_while_signs_play_even_if_holder_is_silent(self):
        feed = self.make(seconds=10)
        feed.final(1, "Ann", "hi")
        self.t = 5.0
        feed.tick()
        self.assertEqual(self.floors(), ["Ann"])

    def test_floor_is_released_after_signs_finish_and_two_seconds_of_silence(self):
        feed = self.make()
        feed.final(1, "Ann", "hi")
        feed.client_message({"type": "ack", "seq": 1, "event": "finished"})
        self.t = 1.9
        feed.tick()
        self.assertEqual(self.floors(), ["Ann"])
        self.t = 2.0
        feed.tick()
        self.assertEqual(self.floors(), ["Ann", None])

    def test_silence_is_measured_from_the_last_caption_activity(self):
        feed = self.make(signs=False)
        feed.interim(1, "Ann", "hi")
        self.t = 1.5
        feed.final(1, "Ann", "hi there")       # the last caption activity
        self.t = 3.0
        feed.tick()
        self.assertEqual(self.floors(), ["Ann"])
        self.t = 3.5
        feed.tick()
        self.assertEqual(self.floors(), ["Ann", None])

    def test_silence_counts_from_the_end_of_sign_playback(self):
        feed = self.make(seconds=10)
        feed.final(1, "Ann", "hi")             # last text at t=0, sign plays until t=10
        self.t = 10.0
        feed.client_message({"type": "ack", "seq": 1, "event": "finished"})
        self.t = 11.9
        feed.tick()
        self.assertEqual(self.floors(), ["Ann"])
        self.t = 12.0
        feed.tick()
        self.assertEqual(self.floors(), ["Ann", None])

    def test_floor_is_kept_while_the_holders_line_is_still_open(self):
        feed = self.make()
        feed.interim(1, "Ann", "hi")
        self.t = 50.0
        feed.tick()
        self.assertEqual(self.floors(), ["Ann"])
        feed.final(1, "Ann", "hi.")            # still signed, never silently dropped
        self.assertEqual([m["line"] for m in self.types("sign")], [1])
        self.t = 52.0
        feed.tick()
        self.assertEqual(self.floors(), ["Ann"])  # signs still playing
        feed.client_message({"type": "ack", "seq": 1, "event": "finished"})
        self.t = 54.0
        feed.tick()
        self.assertEqual(self.floors(), ["Ann", None])

    def test_roles_are_only_kept_for_open_lines(self):
        feed = self.make()
        feed.interim(1, "Ann", "hi")
        feed.interim(2, "Bob", "hmm")
        for i in range(20):                    # pushes both open lines out of the snapshot
            feed.final(100 + i, "Ann", "x")
        feed.final(2, "Bob", "hmm.")
        feed.final(1, "Ann", "hi.")
        self.assertEqual(feed._roles, {})

    def test_side_comment_activity_does_not_hold_the_floor(self):
        feed = self.make(signs=False)
        feed.final(1, "Ann", "hi")
        self.t = 1.5
        feed.interim(2, "Bob", "hmm")
        self.t = 2.0
        feed.tick()
        self.assertEqual(self.floors(), ["Ann", None])

    def test_next_speaker_takes_the_floor_after_release(self):
        feed = self.make()
        feed.final(1, "Ann", "hi")
        self.t = 100.0
        feed.tick()                            # the sign times out
        self.t = 102.0
        feed.tick()                            # then 2.0 s of silence
        feed.final(2, "Bob", "hi")
        self.assertEqual(self.floors(), ["Ann", None, "Bob"])
        self.assertEqual([m["line"] for m in self.types("sign")], [1, 2])

    def test_floor_is_released_without_a_client_once_timeouts_drain_the_queue(self):
        feed = self.make()
        feed.final(1, "Ann", "hi")
        self.t = 1.0 + 1.0         # play + grace: the sign times out
        feed.tick()
        self.assertEqual(self.floors(), ["Ann"])
        self.t += 2.0              # then the silence counts from the sign's end
        feed.tick()
        self.assertEqual(self.floors(), ["Ann", None])

    def test_snapshot_includes_the_floor_holder_and_line_roles(self):
        feed = self.make(signs=False)
        self.assertIsNone(feed.snapshot()["floor"])
        feed.final(1, "Ann", "hi")
        feed.interim(2, "Bob", "yo")
        snap = feed.snapshot()
        self.assertEqual(snap["floor"], "Ann")
        self.assertEqual([l["role"] for l in snap["lines"]], [ROLE_FLOOR, ROLE_SIDE])
        self.t = 100.0
        feed.tick()
        self.assertIsNone(feed.snapshot()["floor"])


if __name__ == "__main__":
    unittest.main()
