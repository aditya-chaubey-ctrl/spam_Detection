"""Text cleaning utilities used for both training and prediction."""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{6,}\d")
NUM_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, replace urls/phones/numbers with tokens, strip punctuation."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()
