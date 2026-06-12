"""Write SHA256 checksums for derived outputs and figure exports."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "code" / "data" / "processed",
    ROOT / "figures" / "paper_figures" / "output" / "svg",
    ROOT / "figures" / "paper_figures" / "output" / "pdf",
    ROOT / "figures" / "paper_figures" / "output" / "png",
]
EXTRA_PATTERNS = [
    ROOT / "code" / "two_axis_closure" / "*.csv",
    ROOT / "code" / "two_axis_closure" / "*.json",
]
OUT = ROOT / "DERIVED_OUTPUT_CHECKSUMS.sha256"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rows: list[str] = []
    for target in TARGETS:
        for path in sorted(p for p in target.rglob("*") if p.is_file()):
            rel = path.relative_to(ROOT).as_posix()
            rows.append(f"{digest(path)}  {rel}")
    for pattern in EXTRA_PATTERNS:
        for path in sorted(pattern.parent.glob(pattern.name)):
            rel = path.relative_to(ROOT).as_posix()
            rows.append(f"{digest(path)}  {rel}")
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
