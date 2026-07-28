"""Language-neutral boundary signal extraction for multilingual subtitle reconstruction.

Language profiles isolate all language‑specific linguistic logic from the
universal boundary decision engine.  The core policy uses only the signals
returned by ``analyse_boundary`` and optionally delegates text‑joining
decisions to the profile.

Each profile holds its own word lists, regexes, verb lists, and
capitalisation rules.  The neutral profile makes no lexical or grammatical
assumptions and is safe for any language.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol


# ───────────────────────────── helpers (language‑neutral) ─────────────────────

_FORMATTING_TAG_RE = re.compile(r"</?[^>]+>|\{\\[^}]+\}")
_STRONG_SENTENCE_END_RE = re.compile(r"""[.!?…]+(?:["'’”»)\]}]+)?$""")
_WORD_RE_UNICODE = re.compile(r"\w(?:[^\W\d_](?:['’][^\W\d_])?)*", re.UNICODE)
_DIALOGUE_DASH_RE = re.compile(r"^\s*(?:--?|[–—]|>>)\s+")
_VOICE_TAG_RE = re.compile(
    r"^\s*(?:<v(?:\s+[^>]*)?>|\[[A-Z][A-Z0-9 ._-]{1,30}\]\s*|"
    r"[A-Z][A-Z0-9 ._-]{1,30}:)"
)
_VOICE_TAG_EXTRACT_RE = re.compile(r"^\s*<v\s+(\S[^>]*)>")
_BRACKET_SPEAKER_RE = re.compile(r"^\s*\[([A-Z][A-Z0-9 ._-]{1,30})\]\s*")
_LABEL_SPEAKER_RE = re.compile(r"^\s*([A-Z][A-Z0-9 ._-]{1,30}):(?:\s|$)")


def visible_text(text: str) -> str:
    """Strip formatting tags and normalise whitespace."""
    return _FORMATTING_TAG_RE.sub("", text).strip()


def ends_strong_sentence(text: str) -> bool:
    """Return True when *text* ends with universal terminal punctuation."""
    return _STRONG_SENTENCE_END_RE.search(visible_text(text)) is not None


def words_from(text: str) -> list[str]:
    """Return the sequence of Unicode word tokens in *text*."""
    return [m.group(0) for m in _WORD_RE_UNICODE.finditer(visible_text(text))]


def has_speaker_marker(text: str) -> bool:
    """Return True when *text* starts with a voice tag, bracket, or label."""
    return bool(_DIALOGUE_DASH_RE.match(text) or _VOICE_TAG_RE.match(text))


def text_contains_question(text: str) -> bool:
    """Return True when visible text ends with ``?`` (universal signal)."""
    return visible_text(text).rstrip(" ""'’\"»)]}").endswith("?")


# ─────────────────────────────── signals ──────────────────────────────────────


@dataclass(frozen=True)
class BoundaryLanguageSignals:
    """Language‑specific continuation and independence signals for one boundary.

    The core policy combines these with gap, model probability, speaker
    structure, and universal punctuation to reach a final decision.
    """

    continuation_score: int = 0
    """Positive integer indicates structural continuation evidence."""

    independent_statements: bool = False
    """True when the left and right cues each appear to be a complete,
    independent statement in the target language."""


# ───────────────────────────── language profile protocol ──────────────────────


class BoundaryLanguageProfile(Protocol):
    """Interface for a language‑specific boundary analysis strategy.

    Every profile exposes a ``code`` (BCP‑47 language tag) and at least an
    ``analyse_boundary`` method.
    """

    code: str

    def analyse_boundary(self, left_text: str, right_text: str) -> BoundaryLanguageSignals:
        """Return continuation and independence signals for one cue boundary."""
        ...

    def text_join_issue(self, left: str, right: str) -> str | None:
        """Return a reason string when joining *left* and *right* could corrupt
        text for this language, or ``None`` when joining is safe."""
        ...


def _normalise_speaker(name: str) -> str:
    """Return a comparison‑safe normalised form of a speaker identity.

    * strips surrounding whitespace
    * applies Unicode casefold
    * normalises to NFC
    """
    return unicodedata.normalize("NFC", name.strip().casefold())


# ──────────────────────── NeutralBoundaryProfile ──────────────────────────────


class NeutralBoundaryProfile:
    """Language‑neutral boundary profile with no lexical or grammatical
    assumptions.

    * No English word lists or finite verbs
    * No capitalisation‑based sentence decisions
    * No lexical continuation overrides
    * Unicode‑safe string handling
    """

    code: str = "und"

    # Shared (language‑neutral) helpers – exposed for composition
    visible_text = staticmethod(visible_text)
    ends_strong_sentence = staticmethod(ends_strong_sentence)
    words_from = staticmethod(words_from)
    text_contains_question = staticmethod(text_contains_question)

    # ── boundary analysis ────────────────────────────────────────────────

    def analyse_boundary(self, left_text: str, right_text: str) -> BoundaryLanguageSignals:
        """Return neutral signals — no lexical continuation or independence.

        The neutral profile provides zero continuation score and never
        claims independent statements based on lexical evidence.
        """
        return BoundaryLanguageSignals(continuation_score=0, independent_statements=False)

    # ── text‑joining safety ──────────────────────────────────────────────

    def text_join_issue(self, left: str, right: str) -> str | None:
        """Neutral joining only checks for empty fragments.

        No language‑specific contraction or apostrophe attachment rules.
        """
        if not left.strip() or not right.strip():
            return "subtitle fragments must contain non-whitespace text"
        return None

    def should_attach_without_space(self, left: str, right: str) -> bool:
        """Return True when the two fragments should not receive a space
        separator in the neutral profile.

        The neutral profile only omits the space for known punctuation and
        bracket characters that are universal typographic conventions.
        """
        return bool(
            not left
            or not right
            or left[-1].isspace()
            or right[0] in _NO_SPACE_BEFORE
            or left[-1] in _NO_SPACE_AFTER
        )

    def __repr__(self) -> str:
        return "NeutralBoundaryProfile()"


_NO_SPACE_BEFORE = frozenset(",.!?;:%)]}»”")
_NO_SPACE_AFTER = frozenset("([{«“")


# ──────────────────────── EnglishBoundaryProfile ──────────────────────────────


class EnglishBoundaryProfile:
    """English‑language boundary profile with full lexical and grammatical analysis.

    Preserves the existing English behaviour exactly.  All word lists,
    finite‑verb regex, question auxiliaries, capitalisation assumptions, and
    contraction rules are isolated here.
    """

    code: str = "en"

    # Shared helpers
    visible_text = staticmethod(visible_text)
    ends_strong_sentence = staticmethod(ends_strong_sentence)
    words_from = staticmethod(words_from)
    text_contains_question = staticmethod(text_contains_question)

    # ── English word lists ───────────────────────────────────────────────

    SUBORDINATE_STARTS: frozenset = frozenset(
        {"although", "because", "if", "since", "though", "unless", "when", "while"}
    )

    CONTINUATION_STARTS: frozenset = frozenset(
        {"and", "as", "because", "but", "for", "if", "nor", "or", "so", "that", "while"}
    )

    CONTINUATION_PRONOUNS: frozenset = frozenset(
        {"her", "his", "its", "my", "our", "their", "the", "this", "that", "your"}
    )

    WH_STARTS: frozenset = frozenset(
        {"how", "what", "when", "where", "which", "who", "whom", "whose", "why"}
    )

    QUESTION_STARTS: frozenset = frozenset({
        "am", "are", "can", "could", "did", "do", "does",
        "had", "has", "have", "how", "is", "may", "must",
        "should", "was", "were", "what", "when", "where",
        "which", "who", "whom", "whose", "why", "will", "would",
    })

    INCOMPLETE_ENDINGS: frozenset = frozenset(
        {"a", "an", "and", "as", "at", "because", "but", "for",
         "if", "of", "or", "than", "to", "with"}
    )

    # ── English regexes ──────────────────────────────────────────────────

    _WORD_RE_ASCII = re.compile(r"[A-Za-z]+(?:['''][A-Za-z]+)?")

    _FINITE_VERB_RE: re.Pattern = re.compile(
        r"\b(?:am|are|can|could|did|do|does|had|has|have|is|may|might|must|"
        r"shall|should|was|were|will|would|won't|can't|don't|doesn't|didn't|"
        r"i'm|you're|he's|she's|it's|we're|they're|i'll|you'll|we'll|they'll|"
        r"think|thinks|thought|suspect|suspects|tell|tells|told|want|wants|"
        r"need|needs|know|knows|knew|escaped|issue|live|die)\b",
        re.IGNORECASE,
    )

    _COMPLEMENT_CONTINUATION_RE: re.Pattern = re.compile(
        r"\b(?:i['’]?ll|i\s+will|i\s+can)\s+"
        r"(?:tell|show|explain)\s+(?:you|him|her|them)\b",
        re.IGNORECASE,
    )

    _COMPARATIVE_CONTINUATION_RE: re.Pattern = re.compile(
        r"^\s*better\s+to\b", re.IGNORECASE,
    )

    _LEADING_CONTRACTION_RE: re.Pattern = re.compile(
        r"^['´‘’](?P<suffix>m|t|ve|re|ll|s|d|em)\b", re.IGNORECASE
    )

    _LAST_WORD_RE: re.Pattern = re.compile(r"(?P<word>[A-Za-z]+)$")

    _INVALID_CONTRACTION_BASES: dict[str, frozenset] = {
        "ve": frozenset({"her", "him", "them", "us", "me", "my", "your", "our", "their", "his", "its"}),
        "re": frozenset({"her", "him", "us", "me", "my", "your", "our", "their", "his", "its", "he", "she", "it"}),
        "ll": frozenset({"her", "him", "them", "us", "me", "my", "your", "our", "their", "his", "its"}),
        "d": frozenset({"her", "him", "them", "us", "me", "my", "your", "our", "their", "his", "its"}),
        "s": frozenset(),
        "em": frozenset(),
    }

    # ── English helper methods ───────────────────────────────────────────

    def _english_words(self, text: str) -> list[str]:
        return [m.group(0) for m in self._WORD_RE_ASCII.finditer(self.visible_text(text))]

    def _starts_new_sentence(self, text: str) -> bool:
        """Return True when *text* starts what looks like a new English sentence."""
        if _VOICE_TAG_RE.match(text):  # universal – shared via module-level
            return True
        first_letter = re.search(r"[A-Za-z]", self.visible_text(text))
        return first_letter is not None and first_letter.group(0).isupper()

    def _looks_like_question(self, text: str) -> bool:
        """Return True when *text* appears to be an English question."""
        if self.text_contains_question(text):
            return True
        eng_words = self._english_words(text)
        if not eng_words:
            return False
        first = eng_words[0].lower()
        return first in self.QUESTION_STARTS or (first == "any" and len(eng_words) > 1)

    def _likely_complete_clause(self, text: str) -> bool:
        """Return True when *text* is likely a complete English clause."""
        visible = self.visible_text(text)
        if self.ends_strong_sentence(visible) or self._looks_like_question(visible):
            return True
        eng_words = self._english_words(visible)
        return len(eng_words) >= 2 and self._FINITE_VERB_RE.search(visible) is not None

    def _appears_incomplete(self, text: str) -> bool:
        """Return weak evidence that a cue is a fragment of a larger English clause."""
        visible = self.visible_text(text)
        eng_words = self._english_words(visible)
        if not eng_words or self.ends_strong_sentence(visible) or self._looks_like_question(visible):
            return False
        first = eng_words[0].casefold()
        last = eng_words[-1].casefold()
        return (
            len(eng_words) == 1
            or last in self.INCOMPLETE_ENDINGS
            or first in self.SUBORDINATE_STARTS
            or re.search(r"[a-z]$", visible) is not None
        )

    # ── English continuation scoring ─────────────────────────────────────

    def _continuation_structure_score(self, left: str, right: str) -> int:
        """Score multi-signal English continuation patterns."""
        left_visible = self.visible_text(left)
        right_visible = self.visible_text(right)
        left_words = self._english_words(left_visible)
        right_words = self._english_words(right_visible)
        if not left_words or not right_words:
            return 0

        first_right = right_words[0].casefold()
        first_left = left_words[0].casefold()
        score = 0
        first_letter = re.search(r"[A-Za-z]", right_visible)
        if first_letter is not None and first_letter.group(0).islower() and len(right_words) >= 2:
            score += 2

        if (
            first_right == "than"
            and len(right_words) >= 3
            and self._COMPARATIVE_CONTINUATION_RE.search(left_visible) is not None
        ):
            score += 3

        if (
            first_left in self.SUBORDINATE_STARTS
            and first_right in self.CONTINUATION_PRONOUNS
            and len(left_words) >= 3
            and len(right_words) >= 3
            and self._likely_complete_clause(right_visible)
        ):
            score += 3

        if (
            first_right in self.WH_STARTS
            and len(right_words) >= 3
            and self._starts_new_sentence(right_visible)
            and self._COMPLEMENT_CONTINUATION_RE.search(left_visible) is not None
        ):
            score += 4

        if (
            first_right in self.CONTINUATION_STARTS
            and not self.ends_strong_sentence(left_visible)
            and len(right_words) >= 2
        ):
            score += 2
        return score

    # ── English independence detection ───────────────────────────────────

    def _looks_like_independent_statements(
        self, left: str, right: str, continuation_score: int
    ) -> bool:
        """Return True when left and right appear to be independent English sentences."""
        if continuation_score:
            return False
        if not (self._likely_complete_clause(left) and self._likely_complete_clause(right)):
            return False

        left_incomplete = self._appears_incomplete(left)
        right_incomplete = self._appears_incomplete(right)
        right_visible = self.visible_text(right)
        right_words = self._english_words(right_visible)
        right_starts_lowercase_question = (
            bool(right_words)
            and right_words[0].casefold() in self.WH_STARTS
            and self.ends_strong_sentence(right_visible)
        )
        if left_incomplete and not (
            self.ends_strong_sentence(right_visible)
            and (self._starts_new_sentence(right_visible) or right_starts_lowercase_question)
        ):
            return False
        if right_incomplete and not self.ends_strong_sentence(right_visible):
            return False

        right_starts_sentence = self._starts_new_sentence(right_visible)
        return right_starts_sentence or right_starts_lowercase_question

    # ── Public API: analyse_boundary ─────────────────────────────────────

    def analyse_boundary(self, left_text: str, right_text: str) -> BoundaryLanguageSignals:
        """Return English lexical continuation and independence signals."""
        continuation_score = self._continuation_structure_score(left_text, right_text)
        independent = self._looks_like_independent_statements(
            left_text, right_text, continuation_score,
        )
        return BoundaryLanguageSignals(
            continuation_score=continuation_score,
            independent_statements=independent,
        )

    # ── English contraction checking ─────────────────────────────────────

    def text_join_issue(self, left: str, right: str) -> str | None:
        """Return a reason when joining would corrupt an English contraction."""
        if not left.strip() or not right.strip():
            return "subtitle fragments must contain non-whitespace text"
        fragment = right.lstrip()
        contraction = self._LEADING_CONTRACTION_RE.match(fragment)
        if contraction is None:
            return None

        preceding_word = self._LAST_WORD_RE.search(left.rstrip())
        if preceding_word is None:
            return "apostrophe-leading contraction has no preceding word"

        base = preceding_word["word"].lower()
        suffix = contraction["suffix"].lower()
        if suffix == "m" and base != "i":
            return f"{contraction.group(0)!r} can only attach safely to 'I'"
        if suffix == "t" and base not in {
            "aren", "can", "couldn", "daren", "didn", "doesn",
            "don", "hadn", "hasn", "haven", "isn", "mightn",
            "mustn", "needn", "shan", "shouldn", "wasn",
            "weren", "won", "wouldn",
        }:
            return f"{contraction.group(0)!r} cannot safely attach to {preceding_word.group(0)!r}"
        if base in self._INVALID_CONTRACTION_BASES.get(suffix, ()):
            return f"{contraction.group(0)!r} cannot safely attach to {preceding_word.group(0)!r}"
        return None

    def should_attach_without_space(self, left: str, right: str) -> bool:
        """Return True when joining without a space is safe for English.

        English allows contraction attachment (e.g. ``'re``) without a space.
        """
        if not left or not right:
            return False
        fragment = right.lstrip()
        contraction = self._LEADING_CONTRACTION_RE.match(fragment)
        if contraction is not None and self.text_join_issue(left, right) is None:
            return True
        return bool(
            left[-1].isspace()
            or right[0] in _NO_SPACE_BEFORE
            or left[-1] in _NO_SPACE_AFTER
        )

    def __repr__(self) -> str:
        return "EnglishBoundaryProfile()"


# ────────────────────────── profile resolution ────────────────────────────────


_KNOWN_PROFILES: dict[str, type[NeutralBoundaryProfile] | type[EnglishBoundaryProfile]] = {
    "en": EnglishBoundaryProfile,
    "und": NeutralBoundaryProfile,
}


def resolve_profile(language_code: str | None) -> BoundaryLanguageProfile:
    """Resolve a BCP‑47 language tag to a ``BoundaryLanguageProfile``.

    * ``None`` or empty → ``EnglishBoundaryProfile`` (backward compatible)
    * ``"en"``, ``"en-US"``, ``"en-GB"``, … → ``EnglishBoundaryProfile``
    * Everything else → ``NeutralBoundaryProfile``

    The returned profile is a fresh instance.
    """
    if not language_code:
        return EnglishBoundaryProfile()

    base = language_code.strip().lower().split("-")[0].split("_")[0]

    base = unicodedata.normalize("NFC", base)

    profile_class = _KNOWN_PROFILES.get(base)
    if profile_class is None:
        return NeutralBoundaryProfile()
    return profile_class()


def normalise_speaker(name: str) -> str:
    """Return a comparison‑safe normalised speaker identity string."""
    return _normalise_speaker(name)
