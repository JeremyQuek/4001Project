"""CHAIR object matching (Rohrbach et al. 2018) with character offsets.

This follows the official CHAIR implementation (github.com/LisaAnne/Hallucination,
utils/chair.py): the same synonym file, the same double-word table, the same
"toilet seat" rule, and the same ground truth (instance annotations plus objects
mentioned in the human captions). Two differences, both needed for this project:

  * we keep the character span of every object mention, so mentions can be mapped
    to the VLM's tokens (official CHAIR only returns words);
  * plural -> singular uses NLTK's WordNet lemmatizer instead of the unmaintained
    `pattern` library, with a small rule-based fallback if WordNet is unavailable.
"""
import re
from dataclasses import dataclass, asdict

# Copied from the official CHAIR code.
COCO_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light", "traffic signal",
    "stop light", "fire hydrant", "stop sign", "parking meter", "suit case", "sports ball",
    "baseball bat", "baseball glove", "tennis racket", "wine glass", "hot dog", "cell phone",
    "mobile phone", "teddy bear", "hair drier", "potted plant", "bow tie", "laptop computer",
    "stove top oven", "home plate", "train track",
]
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]

_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")


@dataclass
class Mention:
    word: str          # surface form as matched (after singularising / double-word merge)
    category: str      # COCO category it maps to
    start: int         # character span in the caption
    end: int
    hallucinated: int = -1  # 1 / 0 once labelled against ground truth

    def to_dict(self):
        return asdict(self)


class _Singularizer:
    def __init__(self):
        self._lemmatize = None
        try:
            from nltk.stem import WordNetLemmatizer
            lem = WordNetLemmatizer()
            lem.lemmatize("dogs")  # raises LookupError if the wordnet data is missing
            self._lemmatize = lambda w: lem.lemmatize(w, "n")
        except Exception:
            self._lemmatize = None

    def __call__(self, w):
        if self._lemmatize is not None:
            return self._lemmatize(w)
        if len(w) > 3 and w.endswith("ies"):
            return w[:-3] + "y"
        if len(w) > 3 and w.endswith(("ses", "xes", "ches", "shes")):
            return w[:-2]
        if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w


class Chair:
    def __init__(self, synonyms_path):
        with open(synonyms_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.inverse_synonym = {}
        for ln in lines:
            syns = [s.strip() for s in ln.split(",") if s.strip()]
            for s in syns:
                self.inverse_synonym[s] = syns[0]
        self.objects = set(self.inverse_synonym)
        self.categories = sorted(set(self.inverse_synonym.values()))

        self.double_words = {w: w for w in COCO_DOUBLE_WORDS}
        for a in ANIMAL_WORDS:
            self.double_words[f"baby {a}"] = a
            self.double_words[f"adult {a}"] = a
        for v in VEHICLE_WORDS:
            self.double_words[f"passenger {v}"] = v
        self.double_words["bow tie"] = "tie"
        self.double_words["toilet seat"] = "toilet"
        self.double_words["wine glas"] = "wine glass"

        self._singular = _Singularizer()

    # ----- matching -----------------------------------------------------------------

    def tokenize(self, text):
        """[(token, start, end)] on the lowercased text."""
        return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text.lower())]

    def find_mentions(self, caption):
        toks = self.tokenize(caption)
        words = [(self._singular(t), s, e) for t, s, e in toks]

        merged = []  # (word, start, end)
        i = 0
        while i < len(words):
            if i + 1 < len(words):
                pair = f"{words[i][0]} {words[i + 1][0]}"
                if pair in self.double_words:
                    merged.append((self.double_words[pair], words[i][1], words[i + 1][2]))
                    i += 2
                    continue
            merged.append(words[i])
            i += 1

        present = {w for w, _, _ in merged}
        if "toilet" in present and "seat" in present:  # "toilet seat" is not a chair
            merged = [m for m in merged if m[0] != "seat"]

        return [Mention(w, self.inverse_synonym[w], s, e) for w, s, e in merged if w in self.objects]

    def category_of(self, coco_name):
        return self.inverse_synonym.get(coco_name, coco_name)

    def ground_truth(self, coco_categories, human_captions):
        """Objects counted as present: instance-annotation categories plus categories
        mentioned in any human caption (as in official CHAIR)."""
        gt = {self.category_of(c) for c in coco_categories}
        for cap in human_captions:
            gt.update(m.category for m in self.find_mentions(cap))
        return gt

    def label(self, caption, gt):
        mentions = self.find_mentions(caption)
        for m in mentions:
            m.hallucinated = int(m.category not in gt)
        return mentions


def chair_metrics(labelled_captions):
    """CHAIR-i and CHAIR-s over a list of per-caption mention lists (mentions carry .hallucinated
    or a 'hallucinated' key). CHAIR-i counts every mention, including repeats, as in the official code."""
    n_mentions = n_halluc = n_caps_with = 0
    for mentions in labelled_captions:
        flags = [m["hallucinated"] if isinstance(m, dict) else m.hallucinated for m in mentions]
        n_mentions += len(flags)
        n_halluc += sum(flags)
        n_caps_with += int(any(flags))
    n_caps = len(labelled_captions)
    return {
        "chair_i": n_halluc / n_mentions if n_mentions else 0.0,
        "chair_s": n_caps_with / n_caps if n_caps else 0.0,
        "n_mentions": n_mentions,
        "n_hallucinated": n_halluc,
        "n_captions": n_caps,
    }
