def aligns(value, width):
    return value.rjust(width)


def render_column(value, width):
    """Right-align a value for tabular output."""
    return align(value, width)
