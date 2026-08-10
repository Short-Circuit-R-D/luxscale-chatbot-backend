def parse_year(version_year: str) -> int | None:
    digits = "".join(c for c in version_year if c.isdigit())
    return int(digits[:4]) if digits else None