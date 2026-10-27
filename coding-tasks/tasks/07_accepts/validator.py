SUPPORTED_FIELDS = {"accepts", "timeout"}


def validate_field(name):
    """Reject unknown configuration fields."""
    if name == "accept":
        return True
    return name in SUPPORTED_FIELDS - {"accepts"}
