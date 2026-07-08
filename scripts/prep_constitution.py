"""One-time prep: convert the raw Constitution .md into clean rag/docs/constitution.txt.

The source .md is a converted PDF: a cover page + a long clickable table of
contents, then the operative provisions (section 1 onward). Its quirk: it puts a
BLANK LINE after every visual line, so blank lines are line WRAPS, not paragraph
breaks. A generic blank-line paragraph rule therefore splits mid-sentence.
Statutes have a better structural signal: section numbers ("7.") and subsection
markers ("(1)", "(2)"). So we flatten all whitespace and re-introduce paragraph
breaks at those markers -- each provision becomes one paragraph, which the
engine's chunker then repacks to the target chunk size.

Usage:
    python scripts/prep_constitution.py <path-to-constitution.md> [start_line]

`start_line` (1-based) is where the operative text begins, so the cover page and
table of contents are dropped. Defaults to the line of "This Constitution is
supreme" if not given.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rag"))
from ingest import _NOISE_LINE, _PAGE_NUM, _IMAGE_LINE  # noqa: E402

OUT = ROOT / "rag" / "docs" / "constitution.txt"


def find_start(lines):
    for i, line in enumerate(lines, start=1):
        if "This Constitution is supreme" in line:
            return i
    return 1


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/prep_constitution.py <constitution.md> [start_line]")
    src = pathlib.Path(sys.argv[1])
    lines = src.read_text(encoding="utf-8").splitlines()
    start = int(sys.argv[2]) if len(sys.argv) > 2 else find_start(lines)

    kept = []
    for line in lines[start - 1:]:
        if _NOISE_LINE.match(line) or _PAGE_NUM.match(line) or _IMAGE_LINE.match(line):
            continue
        line = line.replace("\\.", ".").replace("\\-", "-")
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = line.replace("*", "").strip()
        if line:
            kept.append(line)

    flat = re.sub(r"\s+", " ", " ".join(kept)).strip()
    flat = re.sub(r"\s+(\d{1,3}\.\s+\(1\))", r"\n\n\1", flat)                # section start "N. (1)"
    flat = re.sub(r"(?<=[.;:)])\s+(?=\((?:[2-9]|\d\d+)\)\s)", "\n\n", flat)  # later subsections

    paras = [p.strip() for p in flat.split("\n\n") if p.strip()]
    clean = "\n\n".join(paras)
    header = (
        "Constitution of the Federal Republic of Nigeria, 1999.\n"
        "Source: public interactive edition (Nigerian-Constitution.Com).\n"
        "Contains the operative provisions; the source cover page and table of\n"
        "contents have been removed. Reference text only -- not legal advice."
    )
    OUT.write_text(header + "\n\n" + clean, encoding="utf-8")
    print(f"Wrote {OUT}  ({len(clean):,} chars, {len(paras):,} paragraphs)")


if __name__ == "__main__":
    main()
