"""The shape of a quality gate: a list of checks, and how to print it.

Shared by every phase gate so `PASS`/`WARN`/`FAIL` means the same thing and
exits the same way regardless of which phase produced it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Check:
    name: str
    level: str          # PASS | WARN | FAIL
    detail: str


def format_report(checks: list[Check]) -> str:
    """Aligned check list plus the worst level seen, which is the gate verdict."""
    width = max(len(c.name) for c in checks)
    lines = [f"  [{c.level}] {c.name.ljust(width)}  {c.detail}" for c in checks]
    worst = "FAIL" if any(c.level == "FAIL" for c in checks) else \
            "WARN" if any(c.level == "WARN" for c in checks) else "PASS"
    return "\n".join(lines + ["", f"  gate: {worst}"])
