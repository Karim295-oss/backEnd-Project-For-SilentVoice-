import unittest

from sign_resolver import (WordNetUnavailable, cached_resolver, resolve_sign, resolve_words,
                           wordnet_lemmas)


def motion(value, frames=30):
    return [[float(value)] * 258 for _ in range(frames)]


class ManualSynonymTests(unittest.TestCase):
    def test_gloss_in_the_library_resolves_to_itself_as_a_word(self):
        library = {"happy": motion(1)}

        sign = resolve_sign("happy", library.get, synonyms={}, wordnet_synonyms=lambda w: [])

        self.assertEqual(sign.kind, "word")
        self.assertEqual(sign.motion, motion(1))

    def test_manual_table_synonym_resolves_to_a_library_sign(self):
        library = {"happy": motion(1)}

        sign = resolve_sign("glad", library.get, synonyms={"glad": ["happy"]},
                            wordnet_synonyms=lambda w: [])

        self.assertEqual((sign.kind, sign.motion), ("word", motion(1)))

    def test_wordnet_synonym_resolves_to_a_library_sign(self):
        library = {"big": motion(2)}

        sign = resolve_sign("large", library.get, synonyms={},
                            wordnet_synonyms=lambda w: ["huge", "big"] if w == "large" else [])

        self.assertEqual((sign.kind, sign.motion), ("word", motion(2)))

    def test_manual_synonym_is_preferred_over_wordnet(self):
        library = {"happy": motion(1), "glad_wn": motion(9)}

        sign = resolve_sign("glad", library.get, synonyms={"glad": ["happy"]},
                            wordnet_synonyms=lambda w: ["glad_wn"])

        self.assertEqual(sign.motion, motion(1))


class FingerspellingTests(unittest.TestCase):
    def resolve(self, gloss, library):
        return resolve_sign(gloss, library.get, synonyms={}, wordnet_synonyms=lambda w: [])

    def test_unknown_name_is_fingerspelled_with_10_frames_per_letter(self):
        library = {"j": motion(1), "o": motion(2), "e": motion(3)}

        sign = self.resolve("joe", library)

        self.assertEqual(sign.kind, "fingerspell")
        self.assertEqual(len(sign.motion), 30)
        self.assertEqual([f[0] for f in sign.motion], [1.0] * 10 + [2.0] * 10 + [3.0] * 10)

    def test_letter_motion_is_resampled_from_start_to_end(self):
        ramp = [[float(i)] * 258 for i in range(30)]

        sign = self.resolve("ax", {"a": ramp})

        values = [f[0] for f in sign.motion]
        self.assertEqual(len(values), 10)
        self.assertEqual((values[0], values[-1]), (0.0, 29.0))
        self.assertEqual(values, sorted(values))

    def test_letters_missing_from_the_library_are_skipped(self):
        sign = self.resolve("xjy", {"j": motion(1)})

        self.assertEqual(sign.kind, "fingerspell")
        self.assertEqual(len(sign.motion), 10)

    def test_word_of_only_missing_letters_produces_no_sign(self):
        self.assertIsNone(self.resolve("xyz", {"j": motion(1)}))

    def test_non_letters_are_ignored_and_case_is_folded(self):
        sign = self.resolve("O'Neil-2", {"o": motion(2), "n": motion(4), "e": motion(3),
                                          "i": motion(5), "l": None})

        self.assertEqual(len(sign.motion), 40)


class WordNetUnavailableTests(unittest.TestCase):
    def test_unavailable_wordnet_falls_back_to_fingerspelling(self):
        def offline(word):
            raise WordNetUnavailable()

        sign = resolve_sign("jo", {"j": motion(1), "o": motion(2)}.get, synonyms={},
                            wordnet_synonyms=offline)

        self.assertEqual(sign.kind, "fingerspell")


class CachedResolverTests(unittest.TestCase):
    def test_each_gloss_is_loaded_and_resolved_once(self):
        loads = []

        def load(word):
            loads.append(word)
            return motion(1) if word == "happy" else None

        resolve = cached_resolver(load, synonyms={}, wordnet_synonyms=lambda w: [])
        first, second = resolve("happy"), resolve("HAPPY")

        self.assertEqual(first, second)
        self.assertEqual(loads, ["happy"])

    def test_unresolvable_gloss_is_cached_too(self):
        lookups = []
        resolve = cached_resolver(lambda w: None, synonyms={},
                                  wordnet_synonyms=lambda w: lookups.append(w) or [])

        self.assertIsNone(resolve("xyz"))
        self.assertIsNone(resolve("xyz"))
        self.assertEqual(lookups, ["xyz"])

    def test_result_reached_while_wordnet_was_unavailable_is_not_cached(self):
        library = {"j": motion(1), "o": motion(2), "big": motion(3)}
        state = {"online": False}

        def wordnet(word):
            if not state["online"]:
                raise WordNetUnavailable()
            return ["big"]

        resolve = cached_resolver(library.get, synonyms={}, wordnet_synonyms=wordnet)

        self.assertEqual(resolve("jo").kind, "fingerspell")
        state["online"] = True
        self.assertEqual(resolve("jo").kind, "word")  # WordNet now offers "big"

    def test_letter_signs_are_loaded_once_across_words(self):
        loads = []

        def load(word):
            loads.append(word)
            return motion(1) if len(word) == 1 else None

        resolve = cached_resolver(load, synonyms={}, wordnet_synonyms=lambda w: [])
        resolve("ab")
        resolve("ba")

        self.assertEqual(loads.count("a"), 1)


class ResolveWordsTests(unittest.TestCase):
    def test_returns_resolved_glosses_with_their_motions_and_skips_unsignable_ones(self):
        library = {"happy": motion(1), "j": motion(2)}
        resolve = cached_resolver(library.get, synonyms={"glad": ["happy"]},
                                  wordnet_synonyms=lambda w: [])

        words, sequences = resolve_words(["glad", "j", "jj", "xyz"], resolve)

        self.assertEqual(words, ["glad", "j", "jj"])
        self.assertEqual(sequences["glad"], motion(1))  # keyed by the spoken gloss
        self.assertEqual(len(sequences["jj"]), 20)
        self.assertNotIn("xyz", sequences)


class WordNetLemmaTests(unittest.TestCase):
    def setUp(self):
        try:
            wordnet_lemmas("large")
        except WordNetUnavailable:
            self.skipTest("WordNet is not available")

    def test_common_word_has_single_word_synonyms(self):
        self.assertIn("big", wordnet_lemmas("large"))

    def test_proper_names_have_no_synonyms_so_they_get_fingerspelled(self):
        self.assertEqual(wordnet_lemmas("john"), [])
        self.assertEqual(wordnet_lemmas("mary"), [])


if __name__ == "__main__":
    unittest.main()
