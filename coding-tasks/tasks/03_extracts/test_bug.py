from parser import parse_row


def test_fields():
    assert parse_row("a,b,c") == ["a", "b", "c"]


def test_trim():
    assert parse_row(" x,y ") == ["x", "y"]
