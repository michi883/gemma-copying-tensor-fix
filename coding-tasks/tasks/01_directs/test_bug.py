from routing import resolve_route


def test_direct_route():
    assert resolve_route("directs") == "/home"


def test_fallback():
    assert resolve_route("unknown") == "/404"
