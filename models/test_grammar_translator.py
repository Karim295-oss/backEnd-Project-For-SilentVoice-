import unittest

from grammar_translator import english_to_asl, english_to_asl_words, english_to_gloss_tokens


class GlossTokenTests(unittest.TestCase):
    def test_tokens_come_in_asl_order_with_spans_that_slice_the_source_word(self):
        text = "I don't like coffee"

        tokens = english_to_gloss_tokens(text)

        self.assertEqual(
            [(t.gloss, text[t.start:t.end]) for t in tokens],
            [("coffee", "coffee"), ("i", "I"), ("like", "like"), ("not", "n't")],
        )


class GlossOrderingTests(unittest.TestCase):
    # (English, expected ASL gloss) -- worked from the ordering rules in ADR 0001 / CLAUDE.md:
    # [TIME][OBJ][SUBJ][ADJ][VERB][OTHER][NOT][WH], copulas and articles dropped.
    CASES = [
        ("I don't like coffee", "COFFEE I LIKE NOT"),
        ("This is a book", "BOOK THIS"),  # attr is an object, so it precedes the subject
        ("Tomorrow I will go to the store", "TOMORROW STORE I GO"),  # tense auxiliary dropped
        ("What is your name", "NAME YOUR WHAT"),  # WH goes last
        ("I am happy", "I HAPPY"),  # copula dropped
        ("The dog is big", "DOG BIG"),  # article and copula dropped
    ]

    def test_gloss_string_follows_asl_order(self):
        for text, expected in self.CASES:
            with self.subTest(text):
                self.assertEqual(english_to_asl(text), expected)

    def test_word_list_is_the_same_gloss_in_lowercase(self):
        for text, expected in self.CASES:
            with self.subTest(text):
                self.assertEqual(english_to_asl_words(text), expected.lower().split())

    def test_sentence_with_no_signable_words_gives_empty_outputs(self):
        self.assertEqual(english_to_asl("the"), "(empty)")
        self.assertEqual(english_to_asl_words("the"), [])


class MeetingSpeechTests(unittest.TestCase):
    def test_clauses_keep_their_spoken_order_and_me_stays_me(self):
        words = english_to_asl_words(
            "Can you hear me okay, I think the meeting will start in five minutes"
        )

        self.assertIn("me", words)
        self.assertNotIn("i", words[: words.index("me")])  # "I think..." comes after clause one
        self.assertLess(words.index("hear"), words.index("think"))
        self.assertLess(words.index("me"), words.index("meeting"))

    def test_discourse_fillers_are_removed(self):
        words = english_to_asl_words("So basically we need to finish the report before Friday")

        self.assertNotIn("so", words)
        self.assertNotIn("basically", words)
        self.assertIn("report", words)
        self.assertIn("finish", words)

    def test_phrase_fillers_are_removed_but_real_uses_survive(self):
        filler = english_to_asl_words("Well, you know, I mean, it is kind of hard")
        for word in ("well", "you", "know", "mean", "kind", "of"):
            self.assertNotIn(word, filler)
        self.assertIn("hard", filler)

        self.assertIn("know", english_to_asl_words("Do you know the answer"))

    def test_tense_auxiliary_is_dropped_but_ability_modal_is_kept(self):
        self.assertNotIn("will", english_to_asl_words("I will go home"))
        self.assertNotIn("would", english_to_asl_words("I would go home"))
        self.assertIn("can", english_to_asl_words("I can swim"))

    def test_turn_of_only_fillers_yields_no_glosses(self):
        self.assertEqual(english_to_asl_words("Um, uh, you know, basically, actually"), [])
        self.assertEqual(english_to_gloss_tokens("Well, um, I mean"), [])


if __name__ == "__main__":
    unittest.main()
