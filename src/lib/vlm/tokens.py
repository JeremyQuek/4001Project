"""Token/character alignment helpers (no torch needed, so they are unit-testable on CPU)."""

# Probe positions relative to the first token T of an object mention.
POSITION_OFFSETS = {"post": 0, "pre1": 1, "pre2": 2, "pre3": 3}
MAX_OFFSET = max(POSITION_OFFSETS.values())


def strip_after_stop(ids, stops):
    for k, t in enumerate(ids):
        if t in stops:
            return ids[:k]
    return ids


def token_char_ends(tokenizer, ids):
    """ends[i] = length of decode(ids[:i+1]); token i covers characters [ends[i-1], ends[i])."""
    return [len(tokenizer.decode(ids[: i + 1], skip_special_tokens=True)) for i in range(len(ids))]


def char_to_token(ends, char_idx):
    """Index of the token whose span contains char_idx."""
    for i, e in enumerate(ends):
        if e > char_idx:
            return i
    return len(ends) - 1


def mention_token_spans(ends, mentions):
    """[(first_token, last_token)] for each mention's character span (dicts with start/end)."""
    return [(char_to_token(ends, m["start"]), char_to_token(ends, m["end"] - 1)) for m in mentions]
