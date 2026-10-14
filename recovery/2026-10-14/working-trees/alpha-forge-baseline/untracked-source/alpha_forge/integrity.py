"""Cross-platform hashing for immutable, UTF-8 text artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def canonical_text_sha256(path: Path) -> str:
    """Hash text with line endings normalized to LF.

    Git can materialize a committed text artifact as LF on Unix and CRLF on
    Windows without considering it modified. Integrity hashes must therefore
    cover the canonical text, not the platform-specific checkout bytes.
    """
    text = path.read_bytes().decode("utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
