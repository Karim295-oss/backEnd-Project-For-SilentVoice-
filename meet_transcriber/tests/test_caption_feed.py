import os
import sys
import unittest
from collections import namedtuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from caption_feed import ROLE_FLOOR, CaptionLineFeed  # noqa: E402
from sign_frames import HAND_JOINTS, POSE_JOINTS  # noqa: E402

SKELETON = {"pose": POSE_JOINTS, "hand": HAND_JOINTS}


class CaptionLineFeedTests(unittest.TestCase):
    def make(self, **kw):
        self.sent = []
        return CaptionLineFeed(self.sent.append, **kw)

    def test_interim_emits_an_open_line_message(self):
        feed = self.make()
        feed.interim(7, "Ann", "hello")
        self.assertEqual(self.sent, [
            {"v": 1, "type": "floor", "speaker": "Ann"},
            {"v": 1, "type": "line", "id": 1, "speaker": "Ann", "text": "hello", "final": False,
             "role": ROLE_FLOOR}])

    def test_revisions_keep_the_line_id_and_other_blocks_get_new_ids(self):
        feed = self.make()
        feed.interim(7, "Ann", "hello")
        feed.interim(7, "Ann", "hello there")
        feed.interim(9, "Bob", "yo")
        self.assertEqual([(m["id"], m["text"]) for m in self.sent if m["type"] == "line"],
                         [(1, "hello"), (1, "hello there"), (2, "yo")])

    def test_final_locks_the_line_and_later_text_from_the_block_opens_a_new_one(self):
        feed = self.make()
        feed.interim(7, "Ann", "how are you")
        feed.final(7, "Ann", "How are you?")
        feed.interim(7, "Ann", "I am")
        self.assertEqual([(m["id"], m["text"], m["final"]) for m in self.sent if m["type"] == "line"],
                         [(1, "how are you", False), (1, "How are you?", True), (2, "I am", False)])

    def test_final_without_prior_interim_is_a_complete_line(self):
        feed = self.make()
        feed.final(7, "Ann", "Good morning.")
        feed.final(7, "Ann", "Next.")
        self.assertEqual([(m["id"], m["text"], m["final"]) for m in self.sent if m["type"] == "line"],
                         [(1, "Good morning.", True), (2, "Next.", True)])

    def test_snapshot_holds_the_last_ten_lines_at_their_latest_text(self):
        feed = self.make()
        for i in range(12):
            feed.final(i, "Ann", f"line {i}")
        feed.interim(50, "Bob", "typing")
        feed.interim(50, "Bob", "typing more")
        snap = feed.snapshot()
        self.assertEqual(snap["v"], 1)
        self.assertEqual(snap["type"], "snapshot")
        self.assertEqual([(l["id"], l["text"], l["final"]) for l in snap["lines"]],
                         [(i, f"line {i - 1}", True) for i in range(4, 13)] + [(13, "typing more", False)])

    def test_snapshot_is_empty_before_any_caption(self):
        self.assertEqual(self.make().snapshot(), {"v": 1, "type": "snapshot", "floor": None, "lines": [], "skeleton": SKELETON})


Gloss = namedtuple("Gloss", "gloss start end")


def wrist_motion(x):
    """A 2-frame, 258-feature sign motion whose pose left wrist sits at x."""
    frame = [0.0] * 258
    frame[15 * 4] = x
    return [list(frame), list(frame)]


class SignMessageTests(unittest.TestCase):
    def make(self, glosses, motions):
        self.sent = []
        self.translated = []

        def translate(text):
            self.translated.append(text)
            return glosses

        return CaptionLineFeed(self.sent.append, translate=translate, load_motion=motions.get)

    def signs(self):
        return [m for m in self.sent if m["type"] == "sign"]

    def test_final_line_sends_one_sign_per_gloss_in_gloss_order_linked_to_the_line(self):
        feed = self.make([Gloss("coffee", 7, 13), Gloss("i", 0, 1)],
                         {"coffee": wrist_motion(0.5), "i": wrist_motion(0.25)})
        feed.final(3, "Ann", "I like coffee")

        self.assertEqual(self.translated, ["I like coffee"])
        self.assertEqual([m["type"] for m in self.sent], ["floor", "line", "sign", "sign"])
        first, second = self.signs()
        self.assertEqual({k: v for k, v in first.items() if k != "frames"}, {
            "v": 1, "type": "sign", "seq": 1, "line": 1, "gloss": "coffee", "kind": "word",
            "span": [7, 13], "fps": 30, "speed": 1})
        self.assertEqual((second["seq"], second["gloss"], second["span"]), (2, "i", [0, 1]))
        self.assertEqual(len(first["frames"]), 2)
        self.assertEqual(first["frames"][0]["pose"][9], [0.5, 0, 0])  # left_wrist
        self.assertIsNone(first["frames"][0]["left_hand"])

    def test_resolved_sign_kind_and_frames_are_sent(self):
        self.sent = []
        feed = CaptionLineFeed(self.sent.append, translate=lambda t: [Gloss("joe", 0, 3)],
                               resolve=lambda g: ("fingerspell", wrist_motion(0.5) * 15))
        feed.final(3, "Ann", "Joe")

        (sign,) = self.signs()
        self.assertEqual((sign["gloss"], sign["kind"], len(sign["frames"])), ("joe", "fingerspell", 30))

    def test_resolver_returning_nothing_skips_the_gloss(self):
        self.sent = []
        feed = CaptionLineFeed(self.sent.append, translate=lambda t: [Gloss("xyz", 0, 3)],
                               resolve=lambda g: None)
        feed.final(3, "Ann", "xyz")

        self.assertEqual(self.signs(), [])

    def test_interim_lines_send_no_signs(self):
        feed = self.make([Gloss("i", 0, 1)], {"i": wrist_motion(0.25)})
        feed.interim(3, "Ann", "I")

        self.assertEqual(self.signs(), [])
        self.assertEqual(self.translated, [])

    def test_words_with_no_sign_produce_no_message(self):
        feed = self.make([Gloss("zebra", 0, 5), Gloss("i", 6, 7)], {"i": wrist_motion(0.25)})
        feed.final(3, "Ann", "zebra I")

        self.assertEqual([(m["gloss"], m["seq"]) for m in self.signs()], [("i", 1)])

    def test_sequence_numbers_continue_across_lines(self):
        feed = self.make([Gloss("i", 0, 1)], {"i": wrist_motion(0.25)})
        feed.final(3, "Ann", "I")
        feed.final(3, "Ann", "I")

        self.assertEqual([(m["line"], m["seq"]) for m in self.signs()], [(1, 1), (2, 2)])

    def test_line_locked_by_source_ending_also_becomes_signs(self):
        feed = self.make([Gloss("i", 0, 1)], {"i": wrist_motion(0.25)})
        feed.interim(3, "Ann", "I")
        feed.source_ended(3)

        self.assertEqual([(m["line"], m["gloss"]) for m in self.signs()], [(1, "i")])

    def test_loader_error_skips_only_that_gloss(self):
        def load(gloss):
            if gloss == "bad":
                raise ValueError("wrong shape")
            return wrist_motion(0.25)

        self.sent = []
        feed = CaptionLineFeed(self.sent.append, translate=lambda t: [Gloss("bad", 0, 3), Gloss("i", 4, 5)],
                               load_motion=load)
        with self.assertLogs("caption_feed", "WARNING"):
            feed.final(3, "Ann", "bad I")

        self.assertEqual([m["type"] for m in self.sent], ["floor", "line", "sign"])
        self.assertEqual(self.signs()[0]["gloss"], "i")

    def test_translator_error_still_sends_the_line_without_signs(self):
        def translate(text):
            raise RuntimeError("spacy exploded")

        self.sent = []
        feed = CaptionLineFeed(self.sent.append, translate=translate, load_motion=lambda g: None)
        with self.assertLogs("caption_feed", "WARNING"):
            feed.final(3, "Ann", "hello")
        feed.final(3, "Ann", "again")  # the feed keeps working

        self.assertEqual([m["type"] for m in self.sent], ["floor", "line", "line"])

    def test_motion_with_no_detected_frames_produces_no_message(self):
        feed = self.make([Gloss("empty", 0, 5), Gloss("blank", 6, 11)],
                         {"empty": [], "blank": [[0.0] * 258] * 30})
        feed.final(3, "Ann", "empty blank")

        self.assertEqual(self.signs(), [])

    def test_translate_without_load_motion_sends_lines_only(self):
        self.sent = []
        feed = CaptionLineFeed(self.sent.append, translate=lambda t: [Gloss("i", 0, 1)])
        feed.final(3, "Ann", "I")

        self.assertEqual([m["type"] for m in self.sent], ["floor", "line"])

    def test_snapshot_lists_the_skeleton_joint_names(self):
        skeleton = self.make([], {}).snapshot()["skeleton"]

        self.assertEqual(skeleton["pose"][:2], ["nose", "left_eye"])
        self.assertEqual(len(skeleton["pose"]), 13)
        self.assertEqual(skeleton["hand"][:2], ["wrist", "thumb_cmc"])
        self.assertEqual(len(skeleton["hand"]), 21)


if __name__ == "__main__":
    unittest.main()
