import hashlib
import json

import pandas as pd
import pytest

from intraday_research.stages.io_helpers import (
    canonical_csv_bytes,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_file_or_unavailable,
    write_json,
)


def test_sha256_helpers_hash_bytes_and_files(tmp_path):
    payload = b"abc"
    expected = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert sha256_bytes(payload) == expected
    assert sha256_file(path) == expected
    assert sha256_file_or_unavailable(path) == expected
    assert sha256_file_or_unavailable(tmp_path / "missing.bin") == "unavailable"


def test_canonical_json_bytes_sorts_keys_and_rejects_nan():
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'

    with pytest.raises(ValueError, match="Out of range float"):
        canonical_json_bytes({"bad": float("nan")})


def test_canonical_csv_bytes_preserves_frame_order_and_lineterminators():
    df = pd.DataFrame({"b": [2.0], "a": [1.0]})

    assert canonical_csv_bytes(df) == b"b,a\n2.0,1.0\n"


def test_write_json_creates_parent_and_writes_sorted_pretty_json(tmp_path):
    path = tmp_path / "nested" / "payload.json"

    write_json(path, {"b": 2, "a": 1})

    assert json.loads(path.read_text("utf-8")) == {"a": 1, "b": 2}
    assert path.read_text("utf-8").endswith("\n")


def test_write_json_rejects_nan(tmp_path):
    with pytest.raises(ValueError, match="Out of range float"):
        write_json(tmp_path / "payload.json", {"bad": float("nan")})
