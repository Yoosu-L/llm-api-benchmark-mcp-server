import pytest
from llm_api_benchmark_mcp_server.utils.random_prompt import generate_random_word, generate_random_phrase, MIN_WORD_LENGTH, MAX_WORD_LENGTH


def test_generate_random_word():
    word = generate_random_word()
    assert isinstance(word, str)
    assert MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH


def test_generate_random_phrase():
    phrase = generate_random_phrase(10)
    assert isinstance(phrase, str)
    assert phrase.startswith("Please reply back the following section unchanged: ")
    words = phrase.replace("Please reply back the following section unchanged: ", "").split()
    assert len(words) == 10
