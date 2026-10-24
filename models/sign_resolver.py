"""Resolves an ASL gloss to something signable.

Order: the gloss itself, a manual synonym, a WordNet synonym in the sign library, then
fingerspelling (see resolve_sign).
"""
import string
import time
from itertools import chain
from collections import namedtuple

import numpy as np

ResolvedSign = namedtuple("ResolvedSign", "kind motion")


class WordNetUnavailable(Exception):
    """WordNet could not be loaded (no nltk, or no corpus and no way to download it)."""

LETTER_FRAMES = 10  # ~0.35 s per letter at 30 fps
# Common words the library lacks, mapped to library signs (checked before WordNet).
MANUAL_SYNONYMS = {
    "glad": ["happy"],
    "tiny": ["small"],
    "begin": ["start"],
    "hi": ["hello"],
    "okay": ["ok"],
}


WORDNET_RETRY_SECONDS = 60
_wordnet_failed_at = None


def _load_synsets(word):
    global _wordnet_failed_at
    if _wordnet_failed_at is not None and time.monotonic() - _wordnet_failed_at < WORDNET_RETRY_SECONDS:
        raise WordNetUnavailable()  # don't retry a failed download on every word
    try:
        from nltk.corpus import wordnet
        try:
            return wordnet.synsets(word)
        except LookupError:
            import nltk
            if not nltk.download("wordnet", quiet=True):
                raise WordNetUnavailable()
            return wordnet.synsets(word)
    except WordNetUnavailable:
        _wordnet_failed_at = time.monotonic()
        raise
    except Exception as error:
        _wordnet_failed_at = time.monotonic()
        raise WordNetUnavailable() from error


def wordnet_lemmas(word):
    """Single-word WordNet synonyms of `word`, most common sense first. The WordNet corpus is
    downloaded on first use; raises WordNetUnavailable when it cannot be loaded."""
    synsets = _load_synsets(word)
    if any(name == word.capitalize() for synset in synsets for name in synset.lemma_names()):
        return []  # a proper name ("john" is not "toilet"): fingerspell it instead
    found = []
    for synset in synsets:
        for lemma in synset.lemma_names():
            lemma = lemma.lower()
            if lemma != word and "_" not in lemma and lemma not in found:
                found.append(lemma)
    return found


def resolve_sign(gloss, load_motion, *, synonyms=MANUAL_SYNONYMS, wordnet_synonyms=wordnet_lemmas):
    """`load_motion(word)` returns a sign motion or None."""
    word = gloss.lower()
    candidates = chain([gloss, *synonyms.get(word, [])], _lazy(wordnet_synonyms, word))
    for candidate in candidates:
        motion = load_motion(candidate)
        if motion is not None:
            return ResolvedSign("word", motion)
    return _fingerspell(gloss, load_motion)


def _lazy(lookup, word):
    """Defers the WordNet lookup until the earlier candidates have missed."""
    try:
        yield from lookup(word)
    except WordNetUnavailable:
        return


def _fingerspell(word, load_motion):
    frames = []
    for letter in word.lower():
        if letter not in string.ascii_lowercase:
            continue
        motion = load_motion(letter)
        if motion is not None:
            frames.extend(_resample(motion, LETTER_FRAMES))
    return ResolvedSign("fingerspell", frames) if frames else None


def _resample(motion, count):
    """`count` frames spread evenly from the first to the last frame, linearly interpolated."""
    source = np.asarray(motion, dtype=float)
    positions = np.linspace(0, len(source) - 1, count)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(source) - 1)
    weight = (positions - lower)[:, None]
    return (source[lower] * (1 - weight) + source[upper] * weight).tolist()


def cached_resolver(load_motion, *, synonyms=MANUAL_SYNONYMS, wordnet_synonyms=wordnet_lemmas):
    """`resolve(gloss)` -> ResolvedSign or None, remembering sign files and results.

    A result reached while WordNet was unavailable is not remembered, so the gloss is resolved
    again (and WordNet retried) next time instead of staying fingerspelled for good."""
    motions, signs = {}, {}

    def load(word):
        if word not in motions:
            motions[word] = load_motion(word)
        return motions[word]

    def resolve(gloss):
        key = gloss.lower()
        if key in signs:
            return signs[key]
        wordnet_down = False

        def lookup(word):
            nonlocal wordnet_down
            try:
                return wordnet_synonyms(word)
            except WordNetUnavailable:
                wordnet_down = True
                raise

        sign = resolve_sign(gloss, load, synonyms=synonyms, wordnet_synonyms=lookup)
        if not wordnet_down:
            signs[key] = sign
        return sign

    return resolve


def resolve_words(asl_words, resolve):
    """`(words, sequences)` for the glosses that resolve to a sign, in order. Each sequence is
    keyed by the spoken gloss, even when a synonym or fingerspelling supplied its motion."""
    words, sequences = [], {}
    for word in asl_words:
        sign = resolve(word)
        if sign is not None:
            words.append(word)
            sequences[word] = sign.motion
    return words, sequences
