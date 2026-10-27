from batch import process_batch


def test_batch():
    assert process_batch({"payloads": [{"valid": True}, {"valid": False}]}) == [True, False]


def test_empty_batch():
    assert process_batch({"payloads": []}) == []
