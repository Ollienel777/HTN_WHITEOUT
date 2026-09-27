# Harness profile — Ollienel777/HTN_WHITEOUT (Dogwatch, Hack the North 2026)

A solo hackathon repository, with the hackathon pack (the clock, the spec lap,
milestones, the submission). Ideation runs first with `/harness:ideate`.

```harness
repo: Ollienel777/HTN_WHITEOUT
tracker: github
labels: yes
pr scope: loop-label
pack: hackathon
merge policy: auto
wip limit: 6                # size to the backlog; see START "WIP limit"
round cap: 7
settle rounds: 1
mediums block: no
areas: labels
self-approve cards: yes
gate: python scripts/gate.py   # prints GATE PASS / GATE FAIL / GATE OK lines
branch: type/NUM-slug
ticket line: Closes #NUM
harness paths: hackathon/EVENT.md, hackathon/SPEC.md, hackathon/README.md
status: issue
```

## The product

Dogwatch: four machines (a quadcopter, a fixed-wing, two towers) keep watch
over Bellot Strait and pass the watch between them until the dark ship is found
and held. Built for Dominion Dynamics WHITEOUT, which is **scored, not judged**:
a number decides it. `hackathon/SPEC.md` is the authority.

## What is valuable here

The three-minute demo. The pack's "What wins" section is the rule; add the
event's judging criteria and weights from `hackathon/EVENT.md`.

## The gate

`python scripts/gate.py`: install, lint, format, typecheck, test (`-m "not
slow"`), build, smoke, determinism. It checks that the imported `whiteout`
package resolves inside its own worktree and falls back to a per-worktree
`.venv` otherwise; `WHITEOUT_TRANSPORT` selects the transport. Ruff and mypy
are pinned exactly in `pyproject.toml`.

## Traps

- The repo is public. Nothing that reveals the idea is pushed before
  `hacking starts`; `hackathon/ideation/` and `DECISION.md` stay gitignored
  until the spec PR adds the decision.
- `.gitattributes` must force LF for `*.sh` and `*.mjs`, and mark fixtures
  binary: CRLF conversion on Windows has corrupted a binary fixture.
- A per-worktree environment for the gate: a shared editable install makes
  every worktree's gate test the main checkout's code.
