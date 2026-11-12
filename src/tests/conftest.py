import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))


@pytest.fixture
def chair():
    from lib.data.chair import Chair
    return Chair(SRC / "tests" / "fixtures" / "synonyms_small.txt")
