from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import (
    Any,
)


def at_least_tuple(x: Any) -> tuple[Any, ...]:
    """Convert x to a tuple if it is not already a tuple."""
    return x if isinstance(x, tuple) else (x,)


def generate_filename_from_dict(obj: dict[str, Any], suffix: str = ".pickle") -> Path:
    """Generate a filename from a dictionary."""
    assert all(isinstance(k, str) for k in obj)
    keys = "_".join(obj.keys())
    obj_string = json.dumps(
        obj,
        sort_keys=True,
    )  # Convert the dictionary to a sorted string
    obj_bytes = obj_string.encode()  # Convert the string to bytes

    sha256_hash = hashlib.sha256()
    sha256_hash.update(obj_bytes)
    # Convert the hash to a hexadecimal string for the filename
    str_hash = sha256_hash.hexdigest()
    return Path(f"{keys}__{str_hash}{suffix}")
