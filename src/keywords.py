from src.config import KEYWORDS_FILE


def load_keywords(path=None) -> list[str]:
    """Read keywords.txt and return a list of keyword strings."""
    filepath = path or KEYWORDS_FILE
    keywords = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            keywords.append(line)
    return keywords
