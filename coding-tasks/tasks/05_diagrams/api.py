def make_response(document):
    """Return the public document representation."""
    return {
        "title": document["title"],
        "diagram": document.get("diagrams", []),
    }
