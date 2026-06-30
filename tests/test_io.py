"""Tests for JSONL/JSON I/O utilities."""

import tempfile
from pathlib import Path

from src.utils.io import read_json, read_jsonl, round_floats, write_json, write_jsonl


def test_round_floats_simple():
    assert round_floats(0.123456789, 4) == 0.1235


def test_round_floats_nested():
    data = {"a": 0.123456789, "b": [1.111111, {"c": 2.222222}]}
    rounded = round_floats(data, 4)
    assert rounded["a"] == 0.1235
    assert rounded["b"][0] == 1.1111
    assert rounded["b"][1]["c"] == 2.2222


def test_round_floats_preserves_non_floats():
    data = {"x": 42, "y": "hello", "z": True, "w": None}
    rounded = round_floats(data, 4)
    assert rounded == data


def test_jsonl_roundtrip():
    records = [{"id": "q1", "score": 0.12345}, {"id": "q2", "score": 0.67891}]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    write_jsonl(path, records, round_digits=4)
    loaded = read_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0]["score"] == 0.1235
    assert loaded[1]["score"] == 0.6789
    Path(path).unlink()


def test_json_roundtrip():
    data = {"model_id": "test", "accuracy": 0.12345}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    write_json(path, data)
    loaded = read_json(path)
    assert loaded["model_id"] == "test"
    assert loaded["accuracy"] == 0.12345  # JSON write does not round
    Path(path).unlink()


def test_write_jsonl_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sub" / "dir" / "test.jsonl"
        write_jsonl(path, [{"id": "1"}])
        assert path.exists()
        loaded = read_jsonl(path)
        assert len(loaded) == 1
