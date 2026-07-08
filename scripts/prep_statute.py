"""Prep gazette-style statute .md (converted PDF) into clean, reflowed docs/*.txt.

Same problem as the Constitution: converted-PDF markdown uses blank lines as line
WRAPS, so paragraphs must be rebuilt from statutory structure (section numbers and
subsection markers) rather than blank lines. This generalises that reflow to the
Nigerian legislative drafting styles seen across these acts:

    "7. (1)"     (Constitution)
    "1.-(1)"     (Lagos Tenancy Law)
    "3.—(1)"     (Tax Administration Act, em-dash)
    "1. The ..." / "148. This ..."   (sections without a "(1)" subsection)

Front matter (gazette cover + arrangement-of-sections TOC) is removed by slicing
to the operative start line. Output is committed clean text and passes through the
live normalize() unchanged at build time.
"""
import pathlib
import re
import sys

RAG = pathlib.Path(r"C:\Users\NEW USER\Desktop\naija-law-rag\rag")
sys.path.insert(0, str(RAG))
from ingest import _NOISE_LINE, _PAGE_NUM, _IMAGE_LINE  # noqa: E402

# Break before a section heading: number + "." + optional dash + "(1)" or a capital.
_SECTION = re.compile(r"\s+(?=\d{1,3}\s*\.\s*[—-]?\s*(?:\(1\)|[A-Z]))")
# Break before a later subsection "(2)", "(3)", ... "(10)" (not "(1)", handled above)
# -- but ONLY when it follows clause punctuation, so a parenthetical numeral in
# legal prose ("six (6) months", "three (3) months") is NOT split.
_SUBSEC = re.compile(r"(?<=[.;:)])\s+(?=\((?:[2-9]|\d\d+)\)\s)")


def prep(src: pathlib.Path, start_line: int, out: pathlib.Path, header: str):
    lines = src.read_text(encoding="utf-8").splitlines()[start_line - 1:]

    kept = []
    for line in lines:
        if _NOISE_LINE.match(line) or _PAGE_NUM.match(line) or _IMAGE_LINE.match(line):
            continue
        line = line.replace("\\.", ".").replace("\\-", "-")
        line = line.replace("~~", "")                        # strikethrough (Naira symbol)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = line.replace("*", "").strip()
        if line:
            kept.append(line)

    flat = re.sub(r"\s+", " ", " ".join(kept)).strip()
    # Strip recurring gazette running headers / page tags that the PDF inlined
    # (e.g. "A266", "A259-345", "2025 No. 5", the ALL-CAPS running title). These
    # are page furniture, not statute text. The act's own mixed-case title is left
    # alone so the section-148 citation ("cited as the Nigeria Tax Administration
    # Act, 2025") stays intact.
    flat = re.sub(r"\bA\d{3}\b(?:\s*[-–]\s*\d{2,3})?", " ", flat)
    flat = re.sub(r"NIGERIA TAX ADMINISTRATION ACT,\s*2025(?:\s*ACT NO\.\s*5)?", " ", flat)
    # the running header = act title glued to the gazette footer, in the merged
    # spacings the PDF produced (e.g. "Act, 20252025 No. 5", "Act,2025 No. 5").
    # These are removed; the plain section-148 citation "Act, 2025." is not.
    flat = re.sub(r"Nigeria Tax Administration Act,\s*2025\s*2025 No\.\s*5", " ", flat)
    flat = re.sub(r"Nigeria Tax Administration Act,\s*2025 No\.\s*5", " ", flat)
    flat = re.sub(r"20252025 No\.\s*5", " ", flat)
    flat = re.sub(r"2025 No\.\s*5", " ", flat)
    flat = re.sub(r"\s{2,}", " ", flat).strip()
    flat = _SECTION.sub("\n\n", flat)
    flat = _SUBSEC.sub("\n\n", flat)
    paras = [p.strip() for p in flat.split("\n\n") if p.strip()]
    clean = "\n\n".join(paras)
    out.write_text(header.rstrip() + "\n\n" + clean, encoding="utf-8")

    print(f"Wrote {out.name}: {len(clean):,} chars, {len(paras):,} paragraphs")
    print("  first 3 paragraphs:")
    for p in paras[:3]:
        print("   ", repr(p[:150]))
    print()


MLR = pathlib.Path(r"C:\Users\NEW USER\Desktop\MLRevive")
DOCS = RAG / "docs"

prep(
    MLR / "Tenancy Law 2011.md",
    start_line=77,
    out=DOCS / "lagos_tenancy_law.txt",
    header=(
        "Lagos State Tenancy Law 2011 (Tenancy Law No. 14 of 2011).\n"
        "Source: Lagos State of Nigeria Official Gazette No. 37, 26 August 2011.\n"
        "Verbatim operative provisions; gazette cover and arrangement of sections\n"
        "removed. Reference text only -- not legal advice."
    ),
)

prep(
    MLR / "Approved-Copy-to-Print-NIGERIA-TAX-ADMINISTRATION-ACT-2025.pdf.md",
    start_line=286,
    out=DOCS / "tax_administration_act_2025.txt",
    header=(
        "Nigeria Tax Administration Act, 2025 (Act No. 5 of 2025).\n"
        "Source: Federal Republic of Nigeria Official Gazette No. 117, 26 June 2025.\n"
        "Verbatim operative provisions (procedures for administering the tax laws:\n"
        "registration, returns, assessment, enforcement, offences and penalties).\n"
        "Gazette cover and arrangement of sections removed. Not legal advice."
    ),
)
