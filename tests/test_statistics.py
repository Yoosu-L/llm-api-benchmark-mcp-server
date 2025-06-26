import pytest
from llm_api_benchmark_mcp_server.utils.statistics import calculate_statistics, calculate_ttft_statistics


def test_calculate_statistics_empty():
    stats = calculate_statistics([])
    assert stats["total"] == 0.0
    assert stats["avg"] == 0.0
    assert stats["distribution"]["max"] == 0.0
    assert stats["distribution"]["p50"] == 0.0
    assert stats["distribution"]["p10"] == 0.0
    assert stats["distribution"]["p1"] == 0.0
    assert stats["distribution"]["min"] == 0.0


def test_calculate_statistics_single():
    stats = calculate_statistics([10])
    assert stats["total"] == 10.0
    assert stats["avg"] == 10.0
    assert stats["distribution"]["max"] == 10.0
    assert stats["distribution"]["p50"] == 10.0
    assert stats["distribution"]["p10"] == 10.0
    assert stats["distribution"]["p1"] == 10.0
    assert stats["distribution"]["min"] == 10.0


def test_calculate_statistics_multiple():
    stats = calculate_statistics([10, 20, 30, 40, 50])
    assert stats["total"] == 150.0
    assert stats["avg"] == 30.0
    assert stats["distribution"]["max"] == 50.0
    assert stats["distribution"]["p50"] == 30.0
    assert stats["distribution"]["p10"] == 10.0
    assert stats["distribution"]["p1"] == 10.0
    assert stats["distribution"]["min"] == 10.0


def test_calculate_ttft_statistics_empty():
    stats = calculate_ttft_statistics([])
    assert stats["avg"] == 0.0
    assert stats["distribution"]["min"] == 0.0
    assert stats["distribution"]["p50"] == 0.0
    assert stats["distribution"]["p90"] == 0.0
    assert stats["distribution"]["p99"] == 0.0
    assert stats["distribution"]["max"] == 0.0


def test_calculate_ttft_statistics_single():
    stats = calculate_ttft_statistics([0.5])
    assert stats["avg"] == 0.5
    assert stats["distribution"]["min"] == 0.5
    assert stats["distribution"]["p50"] == 0.5
    assert stats["distribution"]["p90"] == 0.5
    assert stats["distribution"]["p99"] == 0.5
    assert stats["distribution"]["max"] == 0.5


def test_calculate_ttft_statistics_multiple():
    stats = calculate_ttft_statistics([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert stats["avg"] == 0.55
    assert stats["distribution"]["min"] == 0.1
    assert stats["distribution"]["p50"] == 0.5
    assert stats["distribution"]["p90"] == 0.9
    assert stats["distribution"]["p99"] == 1.0
    assert stats["distribution"]["max"] == 1.0
