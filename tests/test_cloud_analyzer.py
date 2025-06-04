import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest

from complete_ai_analyzer import CloudAnalyzer


def test_parse_analysis_response():
    sample_text = (
        "Summary: This document explores AI.\n"
        "It provides general info.\n"
        "\n"
        "Key Points:\n"
        "- AI is transforming industries.\n"
        "- It requires large data.\n"
        "\n"
        "Insights:\n"
        "- Invest in infrastructure.\n"
        "- Focus on ethics.\n"
    )

    analyzer = CloudAnalyzer()
    summary, key_points, insights = analyzer._parse_analysis_response(sample_text)

    assert summary == "This document explores AI. It provides general info."
    assert key_points == [
        "AI is transforming industries.",
        "It requires large data."
    ]
    assert insights == [
        "Invest in infrastructure.",
        "Focus on ethics."
    ]

