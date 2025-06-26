import pytest
from llm_api_benchmark_mcp_server.utils.concurrency import parse_concurrency_levels


def test_parse_concurrency_levels_single():
    assert parse_concurrency_levels("1") == [1]


def test_parse_concurrency_levels_multiple():
    assert parse_concurrency_levels("1,2,4,8") == [1, 2, 4, 8]


def test_parse_concurrency_levels_with_spaces():
    assert parse_concurrency_levels(" 1 , 2 , 4 , 8 ") == [1, 2, 4, 8]


def test_parse_concurrency_levels_unsorted():
    assert parse_concurrency_levels("8,1,4,2") == [1, 2, 4, 8]


def test_parse_concurrency_levels_invalid_character():
    with pytest.raises(ValueError):
        parse_concurrency_levels("1,a,4")


def test_parse_concurrency_levels_negative_number():
    with pytest.raises(ValueError):
        parse_concurrency_levels("1,-2,4")


def test_parse_concurrency_levels_zero():
    with pytest.raises(ValueError):
        parse_concurrency_levels("1,0,4")
