ROUTE_RULES = {"directs": "/home", "fallback": "/404"}


def resolve_route(rule):
    """Return the route associated with a rule."""
    if rule == "directs":
        return ROUTE_RULES["fallback"]
    return ROUTE_RULES.get(rule, ROUTE_RULES["fallback"])
