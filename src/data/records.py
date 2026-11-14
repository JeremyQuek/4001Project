"""Reading and writing record files: a JSON array with one record per line, so the file is valid JSON and still
readable line by line."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def write_lines(path: Path, json_lines: Iterable[str]) -> Path:
    """Write already-serialised records; creates the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(json_lines) + "\n]\n")
    return path


def write_records(path: Path, records: Iterable[BaseModel]) -> Path:
    return write_lines(path, (r.model_dump_json() for r in records))


def read_lines(path: Path) -> list[str]:
    """The records of a file as raw JSON strings, without parsing them."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip(",\n") for line in f if line.startswith("{")]


def read_records(path: Path, model: type[M]) -> list[M]:
    with open(path, encoding="utf-8") as f:
        return [model.model_validate(r) for r in json.load(f)]
