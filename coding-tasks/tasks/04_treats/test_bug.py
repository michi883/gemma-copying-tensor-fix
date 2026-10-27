from rules import Rule, should_apply


def test_enabled():
    assert should_apply(Rule(treats=True)) is True


def test_disabled():
    assert should_apply(Rule(treats=False)) is False
