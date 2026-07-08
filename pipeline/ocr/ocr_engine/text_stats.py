import re


STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "don", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "per", "que", "s", "same", "she", "should", "so", "some",
    "such", "t", "than", "that", "the", "their", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "us", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "you", "your", "yours", "yourself", "yourselves",
    "da", "das", "de", "do", "dos", "em", "na", "nas", "no", "nos",
    "num", "numa", "numas", "nuns", "o", "os", "para", "pelo", "pela",
    "pelos", "pelas", "por", "se", "suas", "um", "uma", "umas", "uns",
})


def tokenize(text: str) -> set[str]:
    """Tokeniza texto preservando caracteres Unicode (PT, ES, FR, etc)."""
    # \w captura letras Unicode (á, é, ã, ç, ñ, etc)
    words = re.findall(r"\w{4,}", text.lower(), re.UNICODE)
    return {w for w in words if w not in STOPWORDS}


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)+\b|\b\d{2,}\b", text))
