class Rule:
    def __init__(self, treats=False):
        self.treats = treats


def should_apply(rule):
    return rule.treat
