from __future__ import annotations

from dataclasses import dataclass


SPECIAL_TOKENS = ("<pad>", "<unk>", "<boundary>")


@dataclass(slots=True)
class VocabularySnapshot:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    pad_id: int
    unk_id: int
    boundary_id: int
    capacity: int

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def token_for_id(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, self.id_to_token[self.unk_id])

    def lookup_token(self, value: str) -> int | None:
        return self.token_to_id.get(value)


class VocabularyRegistry:
    def __init__(self, capacity: int, special_tokens: tuple[str, ...] = SPECIAL_TOKENS) -> None:
        if capacity <= len(special_tokens):
            raise ValueError("Vocabulary capacity must exceed the number of reserved tokens.")
        self.capacity = capacity
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        for token in special_tokens:
            token_id = len(self._token_to_id)
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        self.pad_id = self._token_to_id["<pad>"]
        self.unk_id = self._token_to_id["<unk>"]
        self.boundary_id = self._token_to_id["<boundary>"]
        self._next_id = len(self._token_to_id)

    @property
    def size(self) -> int:
        return len(self._token_to_id)

    def encode_token(self, value: str) -> int:
        if value in self._token_to_id:
            return self._token_to_id[value]
        if self._next_id >= self.capacity:
            return self.unk_id
        token_id = self._next_id
        self._next_id += 1
        self._token_to_id[value] = token_id
        self._id_to_token[token_id] = value
        return token_id

    def snapshot(self) -> VocabularySnapshot:
        return VocabularySnapshot(
            token_to_id=dict(self._token_to_id),
            id_to_token=dict(self._id_to_token),
            pad_id=self.pad_id,
            unk_id=self.unk_id,
            boundary_id=self.boundary_id,
            capacity=self.capacity,
        )
