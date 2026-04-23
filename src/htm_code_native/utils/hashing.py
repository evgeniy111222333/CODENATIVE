from __future__ import annotations

import hashlib


def stable_int_hash(value: str, modulo: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % modulo
