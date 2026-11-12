import os

import pytest

from lib.data.chair import Chair, chair_metrics


def words(mentions):
    return [(m.word, m.category) for m in mentions]


def test_synonyms_and_plurals(chair):
    ms = chair.find_mentions("Two puppies and a man sit near the tables.")
    assert words(ms) == [("puppy", "dog"), ("man", "person"), ("table", "dining table")]


def test_spans_point_at_the_original_text(chair):
    cap = "A Man holding a Cellphone."
    for m in chair.find_mentions(cap):
        assert cap[m.start:m.end].lower().startswith(m.word[:3])


def test_double_words_are_one_mention(chair):
    ms = chair.find_mentions("A teddy bear on a passenger train.")
    assert words(ms) == [("teddy bear", "teddy bear"), ("train", "train")]
    cap = "A teddy bear on a passenger train."
    assert cap[ms[1].start:ms[1].end] == "passenger train"


def test_baby_animal(chair):
    ms = chair.find_mentions("a baby cat")
    assert words(ms) == [("cat", "cat")]


def test_toilet_seat_is_not_a_chair(chair):
    ms = chair.find_mentions("The toilet seat is up.")
    assert words(ms) == [("toilet", "toilet")]
    # but a seat without a toilet is a chair
    assert words(chair.find_mentions("A wooden seat.")) == [("seat", "chair")]


def test_no_substring_matches(chair):
    assert chair.find_mentions("A carpet and a category of things.") == []


def test_ground_truth_uses_annotations_and_human_captions(chair):
    gt = chair.ground_truth({"person"}, ["A dog runs on the grass."])
    assert gt == {"person", "dog"}


def test_label_and_metrics(chair):
    gt = {"person", "dog"}
    a = chair.label("A man walks a dog next to a cat.", gt)
    b = chair.label("A woman.", gt)
    assert [m.hallucinated for m in a] == [0, 0, 1]
    m = chair_metrics([a, b])
    assert m["chair_i"] == pytest.approx(1 / 4)
    assert m["chair_s"] == pytest.approx(1 / 2)


@pytest.mark.skipif(not os.environ.get("CHAIR_SYNONYMS"), reason="set CHAIR_SYNONYMS to the real synonyms.txt")
def test_real_synonym_file_loads():
    c = Chair(os.environ["CHAIR_SYNONYMS"])
    assert len(c.categories) == 80
    assert words(c.find_mentions("a hot dog and a cell phone")) == [("hot dog", "hot dog"), ("cell phone", "cell phone")]
