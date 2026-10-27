import pytest
from validator import validate_field


def test_accepts():
    assert validate_field("accepts") is True


def test_existing_timeout():
    assert validate_field("timeout") is True


def test_unknown():
    assert validate_field("unknown") is False


def test_singular_not_accepted():
    assert validate_field("accept") is False
