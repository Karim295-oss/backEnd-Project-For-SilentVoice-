"""
ASL <-> English Rule-Based Grammar Translator.
Uses spaCy for dependency parsing and POS tagging to reorder sentence structure.
"""
from dataclasses import dataclass

import spacy

# ── Load spaCy English Model ───────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading English language model for spaCy...")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# ── English -> ASL ─────────────────────────────────────────────────────────────
# ASL Grammar Rule: [TIME] [TOPIC/OBJECT] [SUBJECT] [ADJECTIVE] [VERB] [WH-QUESTION]
# - Drop copula "be" (am, is, are, was, were)
# - Drop articles (the, a, an)
# - Drop infinitive "to"
# - Move time expressions to the front
# - Move WH-questions to the end

STOP_WORDS = {"be", "the", "a", "an", "to", "do", "am", "is", "are", "was",
              "were", "at", "in", "of", "very", "just", "have", "has",
              # tense auxiliaries; ability/obligation modals (can, must, should) are kept
              "will", "would", "shall"}

# Spoken fillers and hedges dropped from the gloss. Multi-word ones (you know, I mean,
# kind of, sort of) and context-dependent ones (well, so) are handled in _filler_indices.
FILLER_WORDS = {"um", "uh", "er", "hmm", "basically", "actually", "literally"}

# Dependency labels that start a new clause: the root plus clausal complements, adverbial
# clauses and parataxis. Verb conjuncts are checked separately in _clause_head.
CLAUSE_DEPS = {"ROOT", "ccomp", "advcl", "parataxis"}

TIME_LEXICON = {"tomorrow", "yesterday", "today", "now", "later", "soon",
                "before", "after", "always", "never", "sometimes", "monday",
                "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}

NEGATION_WORDS = {"not", "never", "no", "nobody", "nothing"}


# Output order of the gloss groups: [TIME][OBJ][SUBJ][ADJ][VERB][OTHER][NOT][WH]
GROUP_ORDER = ("time", "obj", "subj", "adj", "verb", "other", "neg", "wh")


@dataclass(frozen=True)
class GlossToken:
    gloss: str  # lowercase
    start: int  # character span of the source English word in the input text
    end: int


def _filler_indices(doc) -> set[int]:
    """Token indices of spoken fillers and hedges that carry no meaning in the gloss."""
    dropped = set()
    n = len(doc)

    def boundary_before(i):
        return i == 0 or doc[i - 1].is_punct or doc[i - 1].i in dropped

    def boundary_after(i):
        return i >= n or doc[i].is_punct

    for token in doc:
        i, word = token.i, token.lower_
        if word in FILLER_WORDS:
            dropped.add(i)
        elif word == "well" and (token.pos_ == "INTJ" or token.dep_ == "intj"
                                 or (boundary_before(i) and boundary_after(i + 1))):
            dropped.add(i)
        elif word == "so" and boundary_before(i):
            dropped.add(i)  # discourse "so" opening a turn or clause, not "so good"
        elif word == "you" and i + 1 < n and doc[i + 1].lower_ == "know" and boundary_after(i + 2):
            dropped.update((i, i + 1))
        elif word == "i" and i + 1 < n and doc[i + 1].lower_ == "mean" and boundary_after(i + 2):
            dropped.update((i, i + 1))
        elif (word in ("kind", "sort") and i + 1 < n and doc[i + 1].lower_ == "of"
              and not (i > 0 and doc[i - 1].pos_ in ("DET", "ADJ", "PRON"))):
            dropped.update((i, i + 1))
    return dropped


def _clause_head(token):
    """The head of the clause a token belongs to (the sentence root or a clausal dependent)."""
    while token.dep_ not in CLAUSE_DEPS and not (token.dep_ == "conj" and token.pos_ in ("VERB", "AUX")):
        if token.head.i == token.i:
            break
        token = token.head
    return token


def english_to_gloss_tokens(sentence: str) -> list[GlossToken]:
    """Translates English to ASL-ordered gloss tokens, each with its source word's span.

    The sentence is split into clauses; each clause is put into ASL order on its own and the
    clauses keep their spoken order.
    """
    doc = nlp(sentence)
    dropped = _filler_indices(doc)
    clauses = {}  # clause head index -> {group name -> tokens}

    for token in doc:
        if token.i in dropped:
            continue
        lemma = token.lemma_.lower()
        if lemma in STOP_WORDS or token.pos_ in ("PUNCT", "SPACE", "DET"):
            continue

        # Pronouns stay as spoken ("me" must not become "i").
        gloss = token.lower_ if token.pos_ == "PRON" else lemma
        if lemma in NEGATION_WORDS or token.dep_ == "neg":
            group, gloss = "neg", "not"  # negation goes at the end in ASL
        elif token.ent_type_ in ("DATE", "TIME") or lemma in TIME_LEXICON:
            group = "time"
        elif token.tag_ in ("WDT", "WP", "WP$", "WRB"):
            group = "wh"
        elif "subj" in token.dep_:
            group = "subj"
        elif "obj" in token.dep_ or token.dep_ in ("attr", "oprd"):
            group = "obj"
        elif token.pos_ == "VERB":
            group = "verb"
        elif token.pos_ == "ADJ":
            group = "adj"
        else:
            group = "other"

        groups = clauses.setdefault(_clause_head(token).i, {name: [] for name in GROUP_ORDER})
        groups[group].append(GlossToken(gloss, token.idx, token.idx + len(token.text)))

    ordered = sorted(
        clauses.values(), key=lambda g: min(t.start for ts in g.values() for t in ts)
    )
    return [t for groups in ordered for name in GROUP_ORDER for t in groups[name]]


def english_to_asl(sentence: str) -> str:
    """Converts a standard English sentence to ASL Gloss order."""
    glosses = english_to_asl_words(sentence)
    return " ".join(glosses).upper() if glosses else "(empty)"


def english_to_asl_words(sentence: str) -> list[str]:
    """Like english_to_asl, but returns the lowercase ASL-ordered word list (for sequence lookup)."""
    return [t.gloss for t in english_to_gloss_tokens(sentence)]


# ── ASL -> English ─────────────────────────────────────────────────────────────
# We lowercase the gloss first, then use spaCy to detect POS and reorder into
# Subject-Verb-Object order with basic "be" verb insertion where needed.

def asl_to_english(asl_gloss: str) -> str:
    """Converts ASL Gloss back to approximate standard English."""
    # Normalize: lowercase for spaCy, but preserve for output
    words = asl_gloss.strip().split()
    
    # Detect negation
    has_negation = any(w.upper() in ("NOT", "NEVER", "NO") for w in words)
    clean_words = [w for w in words if w.upper() not in ("NOT", "NEVER")]
    
    # Detect WH-question
    wh_tags = {"WDT", "WP", "WP$", "WRB"}
    doc = nlp(" ".join(clean_words).lower())

    subjects = []
    verbs = []
    objects = []
    time_words = []
    wh_words = []
    adjectives = []
    others = []

    for token in doc:
        lemma = token.lemma_.lower()

        # Time
        if token.ent_type_ in ("DATE", "TIME") or lemma in TIME_LEXICON:
            time_words.append(token.text)
            continue

        # WH
        if token.tag_ in wh_tags:
            wh_words.append(token.text)
            continue

        # Subject (pronouns like "i", "you", "he")
        if "subj" in token.dep_ or (token.pos_ == "PRON" and token.dep_ in ("nsubj", "nsubjpass")):
            subjects.append(token.text.upper() if token.text.lower() == "i" else token.text)
            continue

        if token.pos_ == "VERB":
            verbs.append(token.lemma_)
            continue

        if token.pos_ == "ADJ":
            adjectives.append(token.text)
            continue

        if token.pos_ in ("NOUN", "PROPN"):
            objects.append(token.text)
            continue

        if token.pos_ not in ("PUNCT", "SPACE"):
            others.append(token.text)

    # Reconstruct English: Subject + Verb + Adjective + Object + Negation + Time
    english_parts = subjects + verbs + adjectives + objects + others

    if has_negation:
        english_parts.append("not")
        
    if time_words:
        english_parts.append(", ".join(time_words))

    sentence = " ".join(english_parts)

    # Handle questions: put WH word at front
    if wh_words:
        sentence = " ".join(wh_words) + " " + sentence + "?"
    elif sentence and not sentence.endswith("?"):
        sentence += "."

    # Capitalize properly
    sentence = sentence.strip().capitalize()
    
    # Fix "i " -> "I "
    import re
    sentence = re.sub(r'\b(i)\b', 'I', sentence)

    if not sentence or sentence == ".":
        return asl_gloss.capitalize()

    return sentence


# ── Interactive CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🤟 ASL Interactive Grammar Translator 🤟")
    print("  [1] English → ASL Gloss")
    print("  [2] ASL Gloss → English")
    print("  Type 'q' to quit.\n")

    while True:
        mode = input("Choose mode (1/2/q): ").strip()

        if mode.lower() in ('q', 'quit', 'exit'):
            print("Goodbye!")
            break

        if mode == "1":
            sentence = input("  Enter English sentence: ").strip()
            if sentence:
                print(f"  → ASL GLOSS: {english_to_asl(sentence)}\n")

        elif mode == "2":
            gloss = input("  Enter ASL Gloss (e.g. TOMORROW STORE I GO): ").strip()
            if gloss:
                print(f"  → ENGLISH: {asl_to_english(gloss)}\n")

        else:
            print("  ⚠ Invalid option. Type 1, 2, or q.\n")
