from layout import render_column


def test_alignment():
    assert render_column("OK", 5) == "   OK"


def test_unchanged_width():
    assert render_column("HELLO", 5) == "HELLO"
