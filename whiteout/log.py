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
  cannot open, and the reader refuses both the bare tokens and any number
  that parses to a non-finite float — ``1e999`` is valid JSON grammar and
  overflows a double — so that reader and writer agree on what is legal.
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

    The writer refuses anything the reader would reject, and it does so by
    running the reader: every payload is passed back through
    :meth:`~whiteout.types.EpisodeRecord.from_dict` before it is encoded, so
    each reader-side rule — unknown or missing fields, wrong types, an
    unrecognised ``cls`` or ``kind`` — is mirrored on the write path without
    being restated here, and a rule added to the reader later is mirrored the
    day it lands. The checks the reader performs outside ``from_dict`` are
    restated: ``schema_version`` must be :data:`SCHEMA_VERSION`, floats must
    be finite (the offender is named by its dotted field path), a record's
    clocks must agree, ``t`` must not run backwards between records, and — a
    rule about the log rather than about any record in it — ``records`` must
    yield at least one, because :func:`iter_episode_log` rejects an empty log
    and so there is no file this writer could produce for zero records that
    any reader would accept. A log this build cannot read back can never be
    produced by it.

    Round-tripping parses every record twice on write, which costs about
    120 ms per 1000 records — a write now costs roughly what reading the
    same log back costs. That is the price of the promise above, and it is
    paid once per episode against a format whose only purpose is to be read.

    The write is atomic. Records are written to a sibling temp file, which
    replaces ``path`` only after the last one succeeds; a rejected record, or
    an empty ``records``, removes the temp file and leaves any existing log
    untouched.
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
    previous: float | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for index, record in enumerate(records, start=1):
                payload = _writable(record, index)
                if previous is not None and record.t < previous:
                    raise EpisodeLogError(
                        index,
                        f"refusing to write t {record.t!r}, which is before "
                        f"the previous record's {previous!r}",
                    )
                previous = record.t
                handle.write(_encode(payload))
                handle.write("\n")
                written += 1
        if written == 0:
            raise EpisodeLogError(1, "refusing to write an empty episode log")
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
    try:
        EpisodeRecord.from_dict(payload)
    except RecordError as exc:
        raise EpisodeLogError(
            index, f"refusing to write a record the reader would reject: {exc}"
        ) from exc
    return payload


def validate_line(line: str, number: int) -> EpisodeRecord:
    """Parse one log line, or raise :class:`EpisodeLogError` naming ``number``.

    Rejects, in order: a line that is not valid JSON — a truncated write is
    the common case, and the bare ``NaN``/``Infinity``/``-Infinity`` tokens
    that Python's ``json`` accepts as an extension are rejected here too — a
    line whose ``schema_version`` is not :data:`SCHEMA_VERSION`, a non-finite
    float, a line with a missing field, an unknown field or a wrongly typed
    value, and a record whose members disagree with its own ``t``.

    The non-finite check is separate from the token rejection because the two
    catch different things. ``parse_constant`` only ever sees the bare
    tokens, but ``1e999`` is valid JSON grammar that overflows a double, so
    ``json.loads`` yields ``inf`` from it without consulting the hook. The
    writer refuses that value, so the reader must too, or a hand-edited or
    foreign-producer log carries an infinity past the gate's validator and
    into a scorer that averages it to ``nan``.
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
    non_finite = _non_finite_field(payload, "record")
    if non_finite is not None:
        raise EpisodeLogError(
            number,
            f"non-finite float at {non_finite}: NaN and Infinity are not JSON",
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

    A log with no records is rejected, at the end of the iteration rather
    than up front, because a file of nothing but blank lines is as empty as a
    0-byte one. The rule lives here so that every reader inherits it: an
    episode truncated to zero by a killed run or a full disk must not read
    clean through :func:`read_episode_log` and score as an episode with no
    ticks.

    The file is read and closed before the first record is yielded, so a
    caller that peeks at one tick and stops does not hold the handle open —
    on Windows that handle blocks the next run from rewriting the log.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    previous: float | None = None
    yielded = 0
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
        yielded += 1
        yield record
    if yielded == 0:
        raise EpisodeLogError(1, "the episode log is empty")


def read_episode_log(path: Path | str) -> list[EpisodeRecord]:
    """Read a whole episode log into memory, validating every line."""
    return list(iter_episode_log(path))


def validate_episode_log(path: Path | str) -> int:
    """Validate the log at ``path`` and return its record count.

    Raises :class:`EpisodeLogError` naming the failing line, or for an empty
    log, which is never a valid episode — that rule lives in
    :func:`iter_episode_log`, so this function and :func:`read_episode_log`
    enforce the same one.
    """
    count = 0
    for _record in iter_episode_log(path):
        count += 1
    return count
