class FieldParser:
    def extracts(self, row):
        return row.strip().split(",")


def parse_row(row):
    parser = FieldParser()
    return parser.extract(row)
