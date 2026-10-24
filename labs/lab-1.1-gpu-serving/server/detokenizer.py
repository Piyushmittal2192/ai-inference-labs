"""Incremental detokenization: token ids in, printable text deltas out.

Decoding one token at a time is wrong. Byte-level BPE tokens can carry half of
a multi-byte UTF-8 character (emoji, accented letters, CJK), and decoding that
half on its own yields U+FFFD replacement characters on the client's screen.

This keeps a short window of recent ids, re-decodes the window on every step,
and only emits text once it is complete. It is the same technique used by the
incremental detokenizers in TGI and vLLM, and costs O(window), not O(sequence).
"""

from __future__ import annotations

from collections.abc import Callable

REPLACEMENT_CHAR = "�"


class IncrementalDetokenizer:
    def __init__(self, decode: Callable[[list[int]], str]) -> None:
        self._decode = decode
        self._ids: list[int] = []
        self._prefix_offset = 0  # start of the window we re-decode
        self._read_offset = 0  # everything before this has already been emitted

    def add(self, token_id: int) -> str:
        """Append one token and return the newly printable text (may be empty)."""
        self._ids.append(token_id)
        prefix_text = self._decode(self._ids[self._prefix_offset : self._read_offset])
        new_text = self._decode(self._ids[self._prefix_offset :])
        if len(new_text) > len(prefix_text) and not new_text.endswith(REPLACEMENT_CHAR):
            delta = new_text[len(prefix_text) :]
            self._prefix_offset = self._read_offset
            self._read_offset = len(self._ids)
            return delta
        return ""

    def flush(self) -> str:
        """Return any text still held back. Call once, after the last token."""
        if self._read_offset == len(self._ids):
            return ""
        prefix_text = self._decode(self._ids[self._prefix_offset : self._read_offset])
        new_text = self._decode(self._ids[self._prefix_offset :])
        self._prefix_offset = self._read_offset = len(self._ids)
        return new_text[len(prefix_text) :]
