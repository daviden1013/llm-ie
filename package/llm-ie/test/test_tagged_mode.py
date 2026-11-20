"""
Offline tests for the "tagged" input mode of FrameExtractors (no LLM server needed).
Run: python -m pytest test/test_tagged_mode.py
"""
import os
import sys
import json
import asyncio
import warnings
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from llm_ie.utils import TextTagger
from llm_ie.extractors import (FrameExtractor, DirectFrameExtractor, ReviewFrameExtractor, BasicFrameExtractor,
                               BasicReviewFrameExtractor, SentenceFrameExtractor, SentenceReviewFrameExtractor)
from llm_ie.chunkers import SeparatorUnitChunker, NoContextChunker
from llm_ie.prompt_editor import PromptEditor


class FakeEngine:
    """Returns pre-defined responses in order and records the messages."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.log = []

    def _next(self, messages):
        self.log.append([dict(m) for m in messages])
        return {"response": self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]}

    def chat(self, messages, verbose=False, messages_logger=None, **kw):
        return self._next(messages)

    async def chat_async(self, messages, messages_logger=None, **kw):
        return self._next(messages)

    def chat_stream(self, messages, **kw):
        yield {"type": "response", "data": self._next(messages)["response"]}


TEXT = "Patient took aspirin daily. Aspirin was stopped.\n\nNo other drugs. Started aspirin again."
#        sentence 1                 sentence 2              sentence 3      sentence 4


# ---------------- TextTagger ----------------
@pytest.mark.parametrize("level", TextTagger.TAG_LEVELS)
def test_tagger_roundtrip(level):
    tagger = TextTagger(level)
    tagged, starts = tagger.tag(TEXT)
    # each tag points at a non-whitespace character
    assert all(not TEXT[s].isspace() for s in starts)
    assert starts == sorted(starts)
    # removing the inserted tags recovers the original text
    restored = tagged
    for i in range(len(starts), 0, -1):
        restored = restored.replace(f"{tagger.format_tag(i)} ", "", 1)
    assert restored == TEXT


def test_tagger_levels():
    assert len(TextTagger("sentence").get_tag_starts(TEXT)) == 4
    assert len(TextTagger("paragraph").get_tag_starts(TEXT)) == 2
    assert len(TextTagger("line").get_tag_starts(TEXT)) == 2  # empty line is not tagged
    assert TextTagger("sentence").tag(TEXT)[0].startswith("<sentence 1> Patient")


def test_tagger_whitespace_and_empty():
    tagger = TextTagger("line")
    assert tagger.tag("") == ("", [])
    assert tagger.tag("   \n\n ") == ("   \n\n ", [])
    tagged, starts = tagger.tag("  a\n   \nb")
    assert tagged == "  <line 1> a\n   \n<line 2> b"


def test_parse_tag_id_and_strip():
    t = TextTagger("sentence")
    assert [t.parse_tag_id(v) for v in [3, "3", "<sentence 3>", "sentence 3", " <3> ", 2.0]] == [3, 3, 3, 3, 3, 2]
    assert [t.parse_tag_id(v) for v in [None, True, "S3", "3-4", 2.5, [3]]] == [None] * 6
    assert t.strip_tags("<sentence 3> aspirin") == "aspirin"
    assert t.has_tag_collision("see <line 2>") and not t.has_tag_collision("BP <120>")


def test_invalid_params():
    with pytest.raises(ValueError):
        TextTagger("token")
    with pytest.raises(ValueError):
        BasicFrameExtractor(FakeEngine(["[]"]), "{{input}}", input_mode="xml")


# ---------------- span resolution ----------------
def _resolve(entities, tag_ids, text=TEXT, **kw):
    ext = BasicFrameExtractor(FakeEngine(["[]"]), "{{input}}", input_mode="tagged", tag_level="sentence")
    spans = ext._find_entity_spans_tagged(text, entities, tag_ids, **kw)
    return [text[s:e] if (s, e) != (None, None) else None for s, e in [sp if sp else (None, None) for sp in spans]], spans


def test_resolve_repeated_entities():
    # "aspirin" appears in sentences 1, 2 (capitalized), and 4
    texts, spans = _resolve(["aspirin", "aspirin", "aspirin"], [4, 1, 2])
    assert spans[0][0] == TEXT.rindex("aspirin")
    assert spans[1][0] == TEXT.index("aspirin")
    assert spans[2][0] == TEXT.index("Aspirin")
    assert texts == ["aspirin", "aspirin", "Aspirin"]


def test_resolve_wrong_or_missing_tag_dropped():
    # "No other drugs" is in sentence 3, not sentence 1 -> must not search whole text
    texts, spans = _resolve(["No other drugs", "aspirin", "aspirin"], [1, None, 99])
    assert spans == [None, None, None]


def test_resolve_repeat_within_segment_in_order():
    text = "Aspirin and aspirin. Then aspirin."
    texts, spans = _resolve(["aspirin", "aspirin", "aspirin"], [1, 1, 1], text=text)
    assert [s[0] for s in spans[:2]] == [0, text.index("aspirin")]
    assert spans[2] is None  # third occurrence is in sentence 2, not after tag 1 within the segment


def test_resolve_spill_over_next_tag():
    # entity starts in sentence 1 but ends in sentence 2
    text = "He took aspirin. Daily dose."
    texts, spans = _resolve(["aspirin. Daily"], [1], text=text)
    assert texts == ["aspirin. Daily"]


def test_resolve_fuzzy_and_tag_in_entity():
    texts, _ = _resolve(["Started aspirin again now"], [4], fuzzy_score_cutoff=0.6)
    assert texts == ["Started aspirin again"]
    texts, _ = _resolve(["<sentence 3> No other drugs"], [3])
    assert texts == ["No other drugs"]
    texts, _ = _resolve(["ASPIRIN"], [2], case_sensitive=True)
    assert texts == [None]


def test_fuzzy_max_start_default_unchanged():
    ext = BasicFrameExtractor(FakeEngine(["[]"]), "{{input}}")
    assert ext._get_closest_substring(TEXT, "took aspirin daily") == ext._get_closest_substring(TEXT, "took aspirin daily", max_start=None)


# ---------------- end-to-end ----------------
RESP = json.dumps([
    {"entity_text": "aspirin", "tag_id": 1, "attr": {"type": "drug"}},
    {"entity_text": "Aspirin", "tag_id": "<sentence 2>", "attr": {"type": "drug"}},
    {"entity_text": "aspirin", "tag_id": 4, "attr": {"type": "drug"}},
    {"entity_text": "aspirin", "attr": {"type": "drug"}},           # missing tag_id -> dropped
])
EXPECTED_STARTS = sorted([TEXT.index("aspirin"), TEXT.index("Aspirin"), TEXT.rindex("aspirin")])


def test_basic_tagged_end_to_end():
    eng = FakeEngine([RESP])
    ext = BasicFrameExtractor(eng, "Extract drugs.\n\n{{input}}", system_prompt="You are a clinical NLP assistant.",
                              input_mode="tagged")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frames = ext.extract_frames(TEXT)
    assert any("tag_id" in str(x.message) for x in w)
    assert sorted(f.start for f in frames) == EXPECTED_STARTS
    assert all(f.attr == {"type": "drug"} and "tag_id" not in f.attr for f in frames)

    sys_msg, user_msg = eng.log[0]
    assert sys_msg["role"] == "system"
    assert sys_msg["content"].startswith("# System instruction")
    assert sys_msg["content"].rstrip().endswith("# Project instruction\nYou are a clinical NLP assistant.")
    assert "{{" not in sys_msg["content"]
    assert user_msg["content"] == "Extract drugs.\n\n" + TextTagger("sentence").tag(TEXT)[0]


def test_tagged_without_custom_system_prompt():
    eng = FakeEngine([RESP])
    ext = BasicFrameExtractor(eng, "{{input}}", input_mode="tagged", tag_level="word")
    ext.extract_frames(TEXT)
    sys_content = eng.log[0][0]["content"]
    assert "Project instruction" not in sys_content and "<word 1>" in sys_content
    assert ext.system_prompt is None  # custom system prompt attribute is untouched


def test_tagged_async_dict_and_stream():
    eng = FakeEngine([RESP])
    ext = BasicFrameExtractor(eng, "Type: {{note_type}}\n{{input}}", input_mode="tagged")
    content = {"input": TEXT, "note_type": "discharge"}
    frames = ext.extract_frames(content, document_key="input", concurrent=True)
    assert sorted(f.start for f in frames) == EXPECTED_STARTS
    assert "<sentence 1> Patient" in eng.log[0][1]["content"] and "Type: discharge" in eng.log[0][1]["content"]

    for _ in ext.stream(content, document_key="input"):
        pass
    assert "<sentence 1> Patient" in eng.log[-1][1]["content"]


def test_tagged_paragraph_units_offsets():
    # units are paragraphs; tags are sentences within each unit. Frame offsets are document-level.
    resp_p1 = json.dumps([{"entity_text": "Aspirin", "tag_id": 2}])
    resp_p2 = json.dumps([{"entity_text": "aspirin", "tag_id": 2}])
    eng = FakeEngine([resp_p1, resp_p2])
    ext = DirectFrameExtractor(eng, unit_chunker=SeparatorUnitChunker("\n\n"), prompt_template="{{input}}",
                               context_chunker=NoContextChunker(), input_mode="tagged")
    frames = ext.extract_frames(TEXT)
    assert [f.start for f in frames] == [TEXT.index("Aspirin"), TEXT.rindex("aspirin")]
    assert eng.log[1][1]["content"].startswith("<sentence 1> No other drugs.")


def test_tagged_review_addition():
    first = json.dumps([{"entity_text": "aspirin", "tag_id": 1}])
    second = json.dumps([{"entity_text": "aspirin", "tag_id": 4}])
    eng = FakeEngine([first, second])
    ext = BasicReviewFrameExtractor(eng, "{{input}}", review_mode="addition", input_mode="tagged")
    frames = ext.extract_frames(TEXT)
    assert sorted(f.start for f in frames) == [TEXT.index("aspirin"), TEXT.rindex("aspirin")]
    # review turn keeps the tagged system instruction
    assert eng.log[1][0]["content"].startswith("# System instruction") and len(eng.log[1]) == 4


def test_sentence_unit_with_sentence_tag_warns():
    with pytest.warns(UserWarning, match="only one tag"):
        SentenceFrameExtractor(FakeEngine(["[]"]), "{{input}}", input_mode="tagged", tag_level="sentence")


# ---------------- prompt guide ----------------
@pytest.mark.parametrize("cls", [DirectFrameExtractor, ReviewFrameExtractor, BasicFrameExtractor,
                                 BasicReviewFrameExtractor, SentenceFrameExtractor, SentenceReviewFrameExtractor])
def test_prompt_guide_modes(cls):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tagged = cls.get_prompt_guide(input_mode="tagged")
        plain = cls.get_prompt_guide()
    assert "tag_id" in tagged and "tag_id" not in plain


def test_prompt_editor_mode_selection():
    ext = BasicFrameExtractor(FakeEngine(["[]"]), "{{input}}", input_mode="tagged")
    assert "tagged input mode" in PromptEditor(None, ext).prompt_guide
    assert "tagged input mode" not in PromptEditor(None, BasicFrameExtractor).prompt_guide
    assert "tagged input mode" in PromptEditor(None, BasicFrameExtractor, input_mode="tagged").prompt_guide
