"""Prep gazette-style statute .md (converted PDF) into a clean, reflowed rag/docs/*.txt.

Same problem as the Constitution: converted-PDF markdown uses blank lines as line
WRAPS, so paragraphs must be rebuilt from statutory structure (section numbers and
subsection markers) rather than blank lines. This handles the Nigerian legislative
drafting styles seen across these acts:

    "7. (1)"     (Constitution)
    "1.-(1)"     (Lagos Tenancy Law)
    "3.—(1)"     (Tax Administration Act, em-dash)
    "1. The ..." / "148. This ..."   (sections without a "(1)" subsection)

Front matter (gazette cover + arrangement-of-sections TOC) is removed by slicing
to the operative start line. Recurring running headers / page tags are stripped.

Usage:
    python scripts/prep_statute.py <source.md> <start_line> <out_basename> [--tax]

  <start_line>    1-based line where the operative text begins (skips cover + TOC)
  <out_basename>  output file name under rag/docs/, e.g. lagos_tenancy_law.txt
  --tax           also strip the Tax Administration Act's gazette running headers

Edit HEADER below to set the provenance banner for the document being prepped.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rag"))
from ingest import _NOISE_LINE, _PAGE_NUM, _IMAGE_LINE  # noqa: E402

DOCS = ROOT / "rag" / "docs"

# Break before a section heading: number + "." + optional dash + "(1)" or a capital.
_SECTION = re.compile(r"\s+(?=\d{1,3}\s*\.\s*[—-]?\s*(?:\(1\)|[A-Z]))")
# Break before a later subsection "(2)".."(10)" ONLY after clause punctuation, so a
# parenthetical numeral in prose ("six (6) months") is not split.
_SUBSEC = re.compile(r"(?<=[.;:)])\s+(?=\((?:[2-9]|\d\d+)\)\s)")

# Set this per document before running.
HEADER = (
    "<Act title and citation>.\n"
    "Source: <official gazette reference>.\n"
    "Verbatim operative provisions; gazette cover and arrangement of sections\n"
    "removed. Reference text only -- not legal advice."
)


def strip_tax_headers(flat: str) -> str:
    """Remove the Tax Administration Act gazette running headers / page tags."""
    flat = re.sub(r"\bA\d{3}\b(?:\s*[-–]\s*\d{2,3})?", " ", flat)
    flat = re.sub(r"NIGERIA TAX ADMINISTRATION ACT,\s*2025(?:\s*ACT NO\.\s*5)?", " ", flat)
    flat = re.sub(r"Nigeria Tax Administration Act,\s*2025\s*2025 No\.\s*5", " ", flat)
    flat = re.sub(r"Nigeria Tax Administration Act,\s*2025 No\.\s*5", " ", flat)
    flat = re.sub(r"20252025 No\.\s*5", " ", flat)
    flat = re.sub(r"2025 No\.\s*5", " ", flat)
    return re.sub(r"\s{2,}", " ", flat).strip()


def main():
    if len(sys.argv) < 4:
        raise SystemExit("usage: python scripts/prep_statute.py <source.md> <start_line> <out_basename> [--tax]")
    src = pathlib.Path(sys.argv[1])
    start_line = int(sys.argv[2])
    out = DOCS / sys.argv[3]
    is_tax = "--tax" in sys.argv[4:]

    lines = src.read_text(encoding="utf-8").splitlines()[start_line - 1:]
    kept = []
    for line in lines:
        if _NOISE_LINE.match(line) or _PAGE_NUM.match(line) or _IMAGE_LINE.match(line):
            continue
        line = line.replace("\\.", ".").replace("\\-", "-").replace("~~", "")
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = line.replace("*", "").strip()
        if line:
            kept.append(line)

    flat = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if is_tax:
        flat = strip_tax_headers(flat)
    flat = _SECTION.sub("\n\n", flat)
    flat = _SUBSEC.sub("\n\n", flat)

    paras = [p.strip() for p in flat.split("\n\n") if p.strip()]
    clean = "\n\n".join(paras)
    out.write_text(HEADER.rstrip() + "\n\n" + clean, encoding="utf-8")
    print(f"Wrote {out}  ({len(clean):,} chars, {len(paras):,} paragraphs)")


if __name__ == "__main__":
    main()
