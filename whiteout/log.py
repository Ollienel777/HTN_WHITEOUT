"""The episode log: a versioned JSONL writer, reader and validator.

``hackathon/SPEC.md`` §4. One :class:`~whiteout.types.EpisodeRecord` per line,
one line per tick, and this file is the only artifact the scorer, the viewer,
the tuner and the ablation harness read.

The format:

- UTF-8, LF line endings, one JSON object per line, trailing newline.
- Keys sorted and floats written by ``json.dumps``' repr, so that two runs at
  the same seed produce byte-identical files. The gate's determinism step
  compares bytes, so the writer may never emit anything ordering-dependent.
- Every line carries ``schema_version``. A log whose lines disagree with
  :data:`SCHEMA_VERSION` is rejected rather than guessed at.

Every rejection names the 1-based line number, because the thing a reader has
in front of them is a file, and "line 4137" is the difference between a fix
and a bisect.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from whiteout.types import EpisodeRecord, RecordError

__all__ = [
    "SCHEMA_VERSION",
    "EpisodeLogError",
    "iter_episode_log",
    "read_episode_log",
    "validate_episode_log",
    "validate_line",
    "write_episode_log",
]

#: The episode log schema version. Bump on any change to the record shape,
#: and expect every artifact under ``artifacts/`` to be regenerated.
SCHEMA_VERSION = 1


class EpisodeLogError(ValueError):
    """A log line is malformed, or is of a version this build cannot read.

    ``line`` is the 1-based line number, and the message is prefixed with it.
    """

    def __init__(self, line: int, reason: str) -> None:
        self.line = line
        self.reason = reason
        super().__init__(f"line {line}: {reason}")


def _encode(record: EpisodeRecord) -> str:
    return json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))


def write_episode_log(path: Path | str, records: Iterable[EpisodeRecord]) -> int:
    """Write ``records`` as JSONL to ``path``; return how many were written.

    Records are written with the version they carry. Writing a record whose
    ``schema_version`` is not :data:`SCHEMA_VERSION` raises, so that a log
    this build cannot read back can never be produced by it in the first
    place.
    """
    out = Path(path)
    if out.parent != Path():
        out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records, start=1):
            if record.schema_version != SCHEMA_VERSION:
                raise EpisodeLogError(
                    index,
                    f"refusing to write schema_version {record.schema_version}, "
                    f"this build writes {SCHEMA_VERSION}",
                )
            handle.write(_encode(record))
            handle.write("\n")
            written += 1
    return written


def validate_line(line: str, number: int) -> EpisodeRecord:
    """Parse one log line, or raise :class:`EpisodeLogError` naming ``number``.

    Rejects, in order: a line that is not valid JSON (a truncated write is
    the common case), a line whose ``schema_version`` is not
    :data:`SCHEMA_VERSION`, and a line with a missing field, an unknown field
    or a wrongly typed value.
    """
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise EpisodeLogError(number, f"not valid JSON: {exc.msg}") from exc
    if isinstance(payload, dict):
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise EpisodeLogError(
                number,
                f"schema_version {version!r} is not supported, expected {SCHEMA_VERSION}",
            )
    try:
        return EpisodeRecord.from_dict(payload)
    except RecordError as exc:
        raise EpisodeLogError(number, str(exc)) from exc


def iter_episode_log(path: Path | str) -> Iterator[EpisodeRecord]:
    """Yield the records of the log at ``path``, one line at a time.

    Blank lines are skipped, so that a log ending in a newline reads cleanly;
    a whitespace-only line is not blank and is rejected as malformed.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip("\r\n") == "":
                continue
            yield validate_line(line, number)


def read_episode_log(path: Path | str) -> list[EpisodeRecord]:
    """Read a whole episode log into memory, validating every line."""
    return list(iter_episode_log(path))


def validate_episode_log(path: Path | str) -> int:
    """Validate the log at ``path`` and return its record count.

    Raises :class:`EpisodeLogError` naming the failing line, or for an empty
    log, which is never a valid episode.
    """
    count = 0
    for _record in iter_episode_log(path):
        count += 1
    if count == 0:
        raise EpisodeLogError(1, "the episode log is empty")
    return count
