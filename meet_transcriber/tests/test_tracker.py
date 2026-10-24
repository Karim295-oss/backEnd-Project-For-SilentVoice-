import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meet_transcriber import CaptionTracker  # noqa: E402


class FakeSink:
    def __init__(self):
        self.events = []

    def final(self, block, speaker, text):
        self.events.append(("final", block, speaker, text))

    def interim(self, block, speaker, text):
        self.events.append(("interim", block, speaker, text))

    def finals(self):
        return [e[3] for e in self.events if e[0] == "final"]


def snap(text, bid=1, speaker="Ann"):
    return [{"id": bid, "speaker": speaker, "text": text}]


class TrackerTests(unittest.TestCase):
    def make(self, **kw):
        self.sink = FakeSink()
        return CaptionTracker(self.sink, idle=0.8, settle=0.3, **kw)

    def test_growth_not_emitted_until_idle(self):
        t = self.make()
        t.update(snap("hello"), now=0)
        t.update(snap("hello there"), now=0.2)
        t.tick(0.7)
        self.assertEqual(self.sink.finals(), [])
        t.tick(1.1)  # 0.9s since last change
        self.assertEqual(self.sink.finals(), ["hello there"])

    def test_no_reemit_after_idle_flush(self):
        t = self.make()
        t.update(snap("hello"), now=0)
        t.tick(1.0)
        t.tick(2.0)
        self.assertEqual(self.sink.finals(), ["hello"])

    def test_growth_after_flush_emits_only_delta(self):
        t = self.make()
        t.update(snap("hello"), now=0)
        t.tick(1.0)
        t.update(snap("hello world"), now=1.5)
        t.tick(2.5)
        self.assertEqual(self.sink.finals(), ["hello", "world"])

    def test_sentence_followed_by_more_text_commits_immediately(self):
        t = self.make()
        t.update(snap("How are you? I am"), now=0)
        self.assertEqual(self.sink.finals(), ["How are you?"])
        t.tick(1.0)
        self.assertEqual(self.sink.finals(), ["How are you?", "I am"])

    def test_trailing_sentence_end_commits_after_settle(self):
        t = self.make()
        t.update(snap("Good morning."), now=0)
        t.tick(0.2)
        self.assertEqual(self.sink.finals(), [])
        t.tick(0.35)
        self.assertEqual(self.sink.finals(), ["Good morning."])

    def test_revision_before_stable_is_never_emitted(self):
        t = self.make()
        t.update(snap("I want too"), now=0)
        t.update(snap("I want to go"), now=0.3)
        t.tick(1.2)
        self.assertEqual(self.sink.finals(), ["I want to go"])

    def test_revision_after_emit_uses_common_prefix(self):
        t = self.make()
        t.update(snap("I want too go"), now=0)
        t.tick(1.0)  # emitted "I want too go"
        t.update(snap("I want to go home"), now=1.1)
        t.tick(2.0)
        # no tail overlap with what was emitted, so the corrected line is re-sent in full
        self.assertEqual(self.sink.finals(), ["I want too go", "I want to go home"])

    def test_block_removal_flushes_remainder(self):
        t = self.make()
        t.update(snap("partial words"), now=0)
        t.update([], now=0.1)
        self.assertEqual(self.sink.finals(), ["partial words"])
        self.assertEqual(t.blocks, {})

    def test_speaker_change_on_same_block_flushes_old_speaker(self):
        t = self.make()
        t.update(snap("from ann", speaker="Ann"), now=0)
        t.update(snap("from bob", speaker="Bob"), now=0.1)
        t.tick(1.0)
        self.assertEqual([(e[2], e[3]) for e in self.sink.events if e[0] == "final"],
                         [("Ann", "from ann"), ("Bob", "from bob")])

    def test_two_speakers_independent(self):
        t = self.make()
        t.update([{"id": 1, "speaker": "Ann", "text": "hi"},
                  {"id": 2, "speaker": "Bob", "text": "yo"}], now=0)
        t.tick(1.0)
        self.assertEqual(sorted(self.sink.finals()), ["hi", "yo"])

    def test_flush_all(self):
        t = self.make()
        t.update(snap("unfinished"), now=0)
        t.flush_all()
        self.assertEqual(self.sink.finals(), ["unfinished"])

    def test_sliding_window_drops_old_words_without_duplicates(self):
        t = self.make()
        t.update(snap("one two three four"), now=0)
        t.tick(1.0)
        t.update(snap("three four five six"), now=1.1)  # window slid: old words gone
        t.tick(2.0)
        self.assertEqual(self.sink.finals(), ["one two three four", "five six"])

    def test_punctuation_revision_is_not_reemitted(self):
        t = self.make()
        t.update(snap("I am"), now=0)
        t.tick(1.0)
        t.update(snap("I am good, I."), now=1.1)
        t.tick(2.0)
        t.update(snap("I am good."), now=2.1)  # shrank back
        t.tick(3.0)
        self.assertEqual(self.sink.finals(), ["I am", "good, I."])

    def test_live_interim_on_change_and_rate_limited(self):
        t = self.make(live=True)
        t.update(snap("a"), now=0)
        t.update(snap("a b"), now=0.05)   # within 100ms: suppressed
        t.update(snap("a b c"), now=0.2)
        interims = [e[3] for e in self.sink.events if e[0] == "interim"]
        self.assertEqual(interims, ["a", "a b c"])
        t.tick(0.25)  # unchanged: no duplicate interim
        self.assertEqual(len([e for e in self.sink.events if e[0] == "interim"]), 2)

    def test_live_interim_excludes_committed_text(self):
        t = self.make(live=True)
        t.update(snap("One two. Three"), now=0)
        interims = [e[3] for e in self.sink.events if e[0] == "interim"]
        self.assertEqual(self.sink.finals(), ["One two."])
        self.assertEqual(interims[-1], "Three")


if __name__ == "__main__":
    unittest.main()
