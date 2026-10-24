import os
import sys
import unittest
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from caption_feed import CaptionLineFeed  # noqa: E402

Gloss = namedtuple("Gloss", "gloss start end")


def motion(frames):
    """A sign motion of `frames` frames (30 fps) with one detected pose value."""
    frame = [0.0] * 258
    frame[0] = 0.1
    return [list(frame) for _ in range(frames)]


def ack(feed, seq, event):
    feed.client_message({"type": "ack", "seq": seq, "event": event})


class SignPacingTests(unittest.TestCase):
    def make(self, durations):
        """Feed whose glosses g0, g1, ... last the given seconds at 1x."""
        self.sent, self.t = [], 0.0
        glosses = [Gloss(f"g{i}", 0, 1) for i in range(len(durations))]
        motions = {f"g{i}": motion(round(d * 30)) for i, d in enumerate(durations)}
        return CaptionLineFeed(self.sent.append, translate=lambda text: glosses,
                               load_motion=motions.get, now=lambda: self.t)

    def sign_seqs(self):
        return [m["seq"] for m in self.sent if m["type"] == "sign"]

    def speeds(self):
        return [m["speed"] for m in self.sent if m["type"] == "sign"]

    def test_never_more_than_two_signs_in_flight(self):
        feed = self.make([1, 1, 1, 1])
        feed.final(1, "Ann", "x")
        self.assertEqual(self.sign_seqs(), [1, 2])

    def test_finished_ack_sends_the_next_sign(self):
        feed = self.make([1, 1, 1, 1])
        feed.final(1, "Ann", "x")
        ack(feed, 1, "finished")
        self.assertEqual(self.sign_seqs(), [1, 2, 3])
        ack(feed, 2, "finished")
        self.assertEqual(self.sign_seqs(), [1, 2, 3, 4])

    def test_started_ack_alone_does_not_advance(self):
        feed = self.make([1, 1, 1])
        feed.final(1, "Ann", "x")
        ack(feed, 1, "started")
        self.assertEqual(self.sign_seqs(), [1, 2])

    def test_acks_for_unknown_or_repeated_seq_are_ignored(self):
        feed = self.make([1, 1, 1, 1])
        feed.final(1, "Ann", "x")
        ack(feed, 99, "finished")
        ack(feed, 3, "finished")  # queued, not yet sent
        self.assertEqual(self.sign_seqs(), [1, 2])
        ack(feed, 1, "finished")
        ack(feed, 1, "finished")
        self.assertEqual(self.sign_seqs(), [1, 2, 3])

    def test_finished_ack_for_a_sign_that_is_not_the_oldest_is_ignored(self):
        feed = self.make([1, 1, 1, 1])
        feed.final(1, "Ann", "x")
        ack(feed, 2, "finished")
        self.assertEqual(self.sign_seqs(), [1, 2])
        ack(feed, 1, "finished")
        self.assertEqual(self.sign_seqs(), [1, 2, 3])

    def test_in_flight_signs_lists_what_a_late_joiner_missed(self):
        feed = self.make([1, 1, 1])
        self.assertEqual(feed.in_flight_signs(), [])
        feed.final(1, "Ann", "x")
        self.assertEqual([m["seq"] for m in feed.in_flight_signs()], [1, 2])
        ack(feed, 1, "finished")
        self.assertEqual([m["seq"] for m in feed.in_flight_signs()], [2, 3])

    def test_tick_alone_finishes_signs_and_sends_the_next(self):
        feed = self.make([0.5] * 3)
        feed.final(1, "Ann", "x")
        self.t = 1.5
        feed.tick()
        self.assertEqual([m["seq"] for m in feed.in_flight_signs()], [2, 3])

    def test_malformed_acks_are_ignored(self):
        feed = self.make([1, 1, 1])
        feed.final(1, "Ann", "x")
        ack(feed, "1", "finished")
        ack(feed, 1, "bogus")
        ack(feed, None, None)
        for junk in (None, [1], "ack", {"type": "bogus", "seq": 1, "event": "finished"},
                     {"seq": 1, "event": "finished"}):
            feed.client_message(junk)
        self.assertEqual(self.sign_seqs(), [1, 2])

    def test_without_a_client_the_queue_drains_by_timeouts(self):
        feed = self.make([0.5] * 4)
        feed.final(1, "Ann", "x")
        self.t = 1.4
        feed.tick()
        self.assertEqual(self.sign_seqs(), [1, 2])
        self.t = 1.5   # sign 1 ends at 0.5, plus 1.0 grace
        feed.tick()
        self.assertEqual(self.sign_seqs(), [1, 2, 3])
        self.t = 2.0   # sign 2 plays 0.5-1.0, plus grace
        feed.tick()
        self.assertEqual(self.sign_seqs(), [1, 2, 3, 4])

    def test_late_started_ack_pushes_the_timeout_back(self):
        feed = self.make([1, 1, 1])
        feed.final(1, "Ann", "x")
        self.t = 1.5
        ack(feed, 1, "started")  # client began late: now ends at 2.5, times out at 3.5
        self.t = 3.4
        feed.tick()
        self.assertEqual(self.sign_seqs(), [1, 2])
        self.t = 3.5
        feed.tick()
        self.assertEqual(self.sign_seqs(), [1, 2, 3])

    def test_repeated_started_acks_do_not_defer_the_timeout(self):
        feed = self.make([1, 1, 1])
        feed.final(1, "Ann", "x")
        self.t = 1.0
        ack(feed, 1, "started")
        self.t = 1.9
        ack(feed, 1, "started")
        ack(feed, True, "finished")  # a bool is not a seq
        self.t = 3.0                # first start 1.0 + 1.0 play + 1.0 grace
        feed.tick()
        self.assertEqual(self.sign_seqs(), [1, 2, 3])

    def test_speed_is_1_for_a_short_backlog(self):
        feed = self.make([1, 1])
        feed.final(1, "Ann", "x")
        self.assertEqual(self.speeds(), [1.0, 1.0])

    def test_speed_is_backlog_over_3_seconds_capped_at_1_75(self):
        feed = self.make([2, 2, 2, 2, 2])
        feed.final(1, "Ann", "x")
        # backlog at dispatch: 10s -> capped; second: 8s -> capped
        self.assertEqual(self.speeds(), [1.75, 1.75])
        ack(feed, 1, "finished")   # remaining 6s -> capped
        ack(feed, 2, "finished")   # remaining 4s -> 4/3
        self.assertEqual(self.speeds()[2:], [1.75, 1.333])

    def test_faster_speed_shortens_the_expected_end(self):
        feed = self.make([3, 3, 3])  # first dispatched at 1.75x: lasts 1.714s
        feed.final(1, "Ann", "x")
        self.t = 2.7   # < 3 * 1/1.75 + 1.0 grace
        feed.tick()
        self.assertEqual(self.sign_seqs(), [1, 2])
        self.t = 2.72
        feed.tick()
        self.assertEqual(self.sign_seqs(), [1, 2, 3])

    def test_no_sign_is_dropped(self):
        feed = self.make([2] * 7)
        feed.final(1, "Ann", "x")
        for _ in range(10):
            self.t += 10
            feed.tick()
        self.assertEqual(self.sign_seqs(), list(range(1, 8)))

    def test_tick_and_ack_without_signs_enabled_do_nothing(self):
        sent = []
        feed = CaptionLineFeed(sent.append)
        feed.tick()
        ack(feed, 1, "finished")
        feed.client_message("junk")
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
