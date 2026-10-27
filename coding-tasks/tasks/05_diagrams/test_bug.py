from api import make_response


def test_diagrams():
    assert make_response({"title": "Plan", "diagrams": ["flow", "tree"]}) == {
        "title": "Plan", "diagrams": ["flow", "tree"]
    }


def test_empty_diagrams():
    assert make_response({"title": "Empty"}) == {"title": "Empty", "diagrams": []}
