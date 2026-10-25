import pytest

from clinicallingo.config import INPUT_LIMITS
from clinicallingo.contracts import RawInput
from clinicallingo.validation import InputValidationError, validate


def test_word_ok():
    r = validate(RawInput(text="  Echogenic ", mode="word"))
    assert r.text == "Echogenic" and r.mode == "word" and r.char_count == 9


@pytest.mark.parametrize("text", ["", "   ", "\x00\x01\x07", "​"])
def test_empty_or_control_only_rejected(text):
    with pytest.raises(InputValidationError) as e:
        validate(RawInput(text=text, mode="word"))
    assert "empty" in e.value.reason


def test_control_chars_stripped_but_text_kept():
    r = validate(RawInput(text="echo\x00genic\x07", mode="word"))
    assert r.text == "echogenic"


def test_bad_mode_rejected():
    with pytest.raises(InputValidationError) as e:
        validate(RawInput(text="echogenic", mode="paragraph"))
    assert e.value.field == "mode"


@pytest.mark.parametrize("mode", ["word", "sentence", "page"])
def test_oversized_rejected(mode):
    with pytest.raises(InputValidationError) as e:
        validate(RawInput(text="a" * (INPUT_LIMITS[mode] + 1), mode=mode))
    assert "limit" in e.value.reason


def test_word_mode_rejects_long_phrases():
    with pytest.raises(InputValidationError):
        validate(RawInput(text="the liver is hypoechoic today", mode="word"))
    assert validate(RawInput(text="common bile duct", mode="word")).text == "common bile duct"


def test_sentence_whitespace_collapsed_and_newlines_removed():
    r = validate(RawInput(text="The   liver\nis  hypoechoic.", mode="sentence"))
    assert r.text == "The liver is hypoechoic."


def test_page_keeps_paragraph_breaks():
    r = validate(RawInput(text="Para one.\n\n\n\nPara two.\tDone.", mode="page"))
    assert r.text == "Para one.\n\nPara two. Done."
