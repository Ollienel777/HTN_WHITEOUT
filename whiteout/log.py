"""The episode log: a versioned JSONL writer, reader and validator.

``hackathon/SPEC.md`` §4. One :class:`~whiteout.types.EpisodeRecord` per line,
one line per tick, and this file is the only artifact the scorer, the viewer,
the tuner and the ablation harness read.

The format:

- UTF-8, LF line endings, one JSON object per line, trailing newline.
- Keys sorted and floats written by ``json.dumps``' repr, so that two runs at
  the same seed produce byte-identical files. The gate's determinism step
  compares bytes, so the writer may never emit anything ordering-dependent.
- **Every value is real JSON.** ``NaN``, ``Infinity`` and ``-Infinity`` are
  Python's extension to JSON, not JSON: the viewer is vanilla JS loaded from
  ``file://`` (§5) and ``JSON.parse`` throws on those tokens, and a ``NaN``
  also makes the advertised round trip silently false, because ``nan != nan``.
  There is no encoding of them that survives both, so the writer refuses a
  non-finite float (naming the field) rather than writing a log the viewer
  cannot open, and the reader refuses the bare tokens rather than parsing
  what it says is not JSON.
- Every line carries ``schema_version``. A log whose lines disagree with
  :data:`SCHEMA_VERSION` is rejected rather than guessed at.
- A record's clocks agree: ``observation.t``, ``intent.t``,
  ``belief_digest.t`` and ``truth.t`` equal the record's ``t``, and ``t`` does
  not run backwards down the file.

Writing is atomic: records go to a sibling temp file that replaces the
destination only once every record has been written. A rejected record can
never leave a truncated log in place of a good one — half an episode that
validates clean is worse than a write that fails, because the scorer and the
tuner would score it as a whole one.

Every rejection names the 1-based line number, because the thing a reader has
in front of them is a file, and "line 4137" is the difference between a fix
and a bisect.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

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

#: Members of a record that carry their own copy of the tick's clock.
_CLOCK_MEMBERS = ("observation", "intent", "belief_digest", "truth")


class EpisodeLogError(ValueError):
    """A log line is malformed, or is of a version this build cannot read.

    ``line`` is the 1-based line number, and the message is prefixed with it.
    """

    def __init__(self, line: int, reason: str) -> None:
        self.line = line
        self.reason = reason
        super().__init__(f"line {line}: {reason}")


def _reject_constant(token: str) -> float:
    raise ValueError(f"{token} is not valid JSON")


def _non_finite_field(value: Any, path: str) -> str | None:
    """Return the dotted path of the first non-finite float, or ``None``."""
    if isinstance(value, float):
        return None if math.isfinite(value) else path
    if isinstance(value, dict):
        for key, item in value.items():
            found = _non_finite_field(item, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _non_finite_field(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _clock_fault(record: EpisodeRecord) -> str | None:
    """Return why a record's clocks disagree, or ``None`` if they agree."""
    for name in _CLOCK_MEMBERS:
        member_t = getattr(record, name).t
        if member_t != record.t:
            return f"record.{name}.t is {member_t!r} but record.t is {record.t!r}"
    return None


def _encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_episode_log(path: Path | str, records: Iterable[EpisodeRecord]) -> int:
    """Write ``records`` as JSONL to ``path``; return how many were written.

    The writer refuses anything the reader would reject: a record whose
    ``schema_version`` is not :data:`SCHEMA_VERSION`, a non-finite float
    (named by its dotted field path), or a record whose clocks disagree. A
    log this build cannot read back can never be produced by it.

    The write is atomic. Records are written to a sibling temp file, which
    replaces ``path`` only after the last one succeeds; a rejected record
    removes the temp file and leaves any existing log untouched.
    """
    out = Path(path)
    directory = out.parent if str(out.parent) else Path()
    if str(directory):
        directory.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(directory) if str(directory) else None,
        prefix=f"{out.name}.",
        suffix=".tmp",
    )
    temp = Path(temp_name)
    written = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for index, record in enumerate(records, start=1):
                handle.write(_encode(_writable(record, index)))
                handle.write("\n")
                written += 1
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    os.replace(temp, out)
    return written


def _writable(record: EpisodeRecord, index: int) -> dict[str, Any]:
    """Return ``record`` as a JSON payload, or raise naming its position."""
    if record.schema_version != SCHEMA_VERSION:
        raise EpisodeLogError(
            index,
            f"refusing to write schema_version {record.schema_version}, "
            f"this build writes {SCHEMA_VERSION}",
        )
    fault = _clock_fault(record)
    if fault is not None:
        raise EpisodeLogError(index, f"refusing to write a record whose clocks disagree: {fault}")
    payload = record.to_dict()
    non_finite = _non_finite_field(payload, "record")
    if non_finite is not None:
        raise EpisodeLogError(
            index,
            f"refusing to write a non-finite float at {non_finite}: NaN and Infinity are not JSON",
        )
    return payload


def validate_line(line: str, number: int) -> EpisodeRecord:
    """Parse one log line, or raise :class:`EpisodeLogError` naming ``number``.

    Rejects, in order: a line that is not valid JSON — a truncated write is
    the common case, and the bare ``NaN``/``Infinity``/``-Infinity`` tokens
    that Python's ``json`` accepts as an extension are rejected here too — a
    line whose ``schema_version`` is not :data:`SCHEMA_VERSION`, a line with a
    missing field, an unknown field or a wrongly typed value, and a record
    whose members disagree with its own ``t``.
    """
    try:
        payload = json.loads(line, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise EpisodeLogError(number, f"not valid JSON: {exc.msg}") from exc
    except ValueError as exc:
        raise EpisodeLogError(number, f"not valid JSON: {exc}") from exc
    if isinstance(payload, dict):
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise EpisodeLogError(
                number,
                f"schema_version {version!r} is not supported, expected {SCHEMA_VERSION}",
            )
    try:
        record = EpisodeRecord.from_dict(payload)
    except RecordError as exc:
        raise EpisodeLogError(number, str(exc)) from exc
    fault = _clock_fault(record)
    if fault is not None:
        raise EpisodeLogError(number, fault)
    return record


def iter_episode_log(path: Path | str) -> Iterator[EpisodeRecord]:
    """Yield the records of the log at ``path``, one line at a time.

    Blank lines are skipped, so that a log ending in a newline reads cleanly;
    a whitespace-only line is not blank and is rejected as malformed. ``t``
    must not run backwards between lines.

    The file is read and closed before the first record is yielded, so a
    caller that peeks at one tick and stops does not hold the handle open —
    on Windows that handle blocks the next run from rewriting the log.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    previous: float | None = None
    for number, line in enumerate(lines, start=1):
        if line.strip("\r\n") == "":
            continue
        record = validate_line(line, number)
        if previous is not None and record.t < previous:
            raise EpisodeLogError(
                number,
                f"t {record.t!r} is before the previous line's {previous!r}",
            )
        previous = record.t
        yield record


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
