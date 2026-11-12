import numpy as np

from lib.vlm.activations import mention_features, pack_shard
from lib.vlm.tokens import MAX_OFFSET, char_to_token, mention_token_spans, strip_after_stop, token_char_ends


class FakeTokenizer:
    """Each id maps to a fixed string, like a BPE vocab with leading-space tokens."""

    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab[i] for i in ids)


VOCAB = {0: "A", 1: " man", 2: " with", 3: " a", 4: " car", 5: "pet", 6: " dog", 7: ".", 8: " teddy", 9: " bear"}
TOK = FakeTokenizer(VOCAB)


def test_char_alignment():
    ids = [0, 1, 2, 3, 6, 7]  # "A man with a dog."
    ends = token_char_ends(TOK, ids)
    assert ends == [1, 5, 10, 12, 16, 17]
    text = TOK.decode(ids)
    assert char_to_token(ends, text.index("dog")) == 4
    assert char_to_token(ends, text.index("man")) == 1


def test_multi_token_object_and_carpet(chair):
    ids = [0, 3, 4, 5, 2, 3, 8, 9, 7]  # "A a carpet with a teddy bear."
    text = TOK.decode(ids)
    ms = [m.to_dict() for m in chair.find_mentions(text)]
    assert [m["word"] for m in ms] == ["teddy bear"]  # "car" inside "carpet" is not matched
    assert mention_token_spans(token_char_ends(TOK, ids), ms) == [(6, 7)]


def test_strip_after_stop():
    assert strip_after_stop([5, 6, 99, 7], {99}) == [5, 6]


def test_mention_features_positions(chair):
    ids = [1, 2, 3, 6, 7]  # " man with a dog."
    text = TOK.decode(ids)
    ms = [m.to_dict() for m in chair.find_mentions(text)]
    G, L, D = len(ids), 2, 4
    # hidden[:, MAX_OFFSET + i] holds caption token i; fill each position with its own index.
    hidden = np.zeros((L, MAX_OFFSET + G, D), np.float16)
    for p in range(MAX_OFFSET + G):
        hidden[:, p] = p - MAX_OFFSET
    tf = {"hidden": hidden, "prob": np.linspace(0.1, 0.9, G), "entropy": np.ones(G)}
    positions = ["post", "pre1", "pre2", "pre3"]
    feats, valid, overlap, stats = mention_features(TOK, ids, ms, tf, positions)

    # mention 0 "man" is caption token 0: pre1 is the last prompt token (valid), pre2/pre3 are in the prompt.
    assert valid["post"][0] and valid["pre1"][0] and not valid["pre2"][0] and not valid["pre3"][0]
    assert feats["pre1"][0][0, 0] == -1
    # mention 1 "dog" is token 3: post reads index 3, pre3 reads token 0, which belongs to "man".
    assert feats["post"][1][0, 0] == 3 and feats["pre2"][1][0, 0] == 1
    assert overlap["pre3"][1] and not overlap["pre2"][1]
    assert stats[1]["token_idx"] == 3 and stats[1]["token_prob"] == tf["prob"][3]

    shard = pack_shard([4, 8], positions, ["u0", "u1"], feats, valid, overlap)
    assert shard["L8_post"].shape == (2, D)
    assert shard["valid_pre2"].tolist() == [False, True]
