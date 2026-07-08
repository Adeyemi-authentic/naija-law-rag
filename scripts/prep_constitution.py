"""One-time prep: convert the raw Constitution .md into clean docs/constitution.txt.

The source .md is a converted PDF: a cover page + a long clickable table of
contents (lines 1-1030), then the operative provisions (section 1 onward). Its
quirk: it puts a BLANK LINE after every visual line, so blank lines are line
WRAPS, not paragraph breaks. A generic blank-line paragraph rule therefore
splits mid-sentence. Statutes have a better structural signal: section numbers
("7.") and subsection markers ("(1)", "(2)"). So we flatten all whitespace and
re-introduce paragraph breaks at those markers -- each provision becomes one
paragraph, which the engine's chunker then repacks to the target chunk size.

The output is committed clean text (inspectable, and what a legal corpus should
be), and it passes through the live normalize() unchanged at build time.
"""
import pathlib
import re
import sys

RAG = pathlib.Path(r"C:\Users\NEW USER\Desktop\naija-law-rag\rag")
sys.path.insert(0, str(RAG))
from ingest import _NOISE_LINE, _PAGE_NUM, _IMAGE_LINE  # noqa: E402

SRC = pathlib.Path(r"C:\Users\NEW USER\Desktop\MLRevive\constitution-of-the-federal-republic-of-nigeria.md")
OUT = RAG / "docs" / "constitution.txt"

lines = SRC.read_text(encoding="utf-8").splitlines()
# Operative text (section 1, "This Constitution is supreme") begins at line 1031.
body_lines = lines[1030:]

# Drop page furniture / image blobs, strip markdown emphasis + escapes, keep text.
kept = []
for line in body_lines:
    if _NOISE_LINE.match(line) or _PAGE_NUM.match(line) or _IMAGE_LINE.match(line):
        continue
    line = line.replace("\\.", ".").replace("\\-", "-")
    line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
    line = re.sub(r"\*(.*?)\*", r"\1", line)
    line = line.replace("*", "").strip()
    if line:
        kept.append(line)

# Flatten wraps: every run of whitespace becomes a single space.
flat = re.sub(r"\s+", " ", " ".join(kept)).strip()

# Re-introduce paragraph breaks at statutory markers.
#  A) a section start "7. (1)" -- keep the section number attached to its text.
flat = re.sub(r"\s+(\d{1,3}\.\s+\(1\))", r"\n\n\1", flat)
#  B) subsequent subsections "(2)", "(3)", ... (not "(1)", handled by A; not a
#     mid-sentence cross-reference like "subsection (2)", which lacks the number).
flat = re.sub(r"\s+(\((?:[2-9]|\d\d+)\)\s)", r"\n\n\1", flat)

paras = [p.strip() for p in flat.split("\n\n") if p.strip()]
clean = "\n\n".join(paras)

header = (
    "Constitution of the Federal Republic of Nigeria, 1999.\n"
    "Source: public interactive edition (Nigerian-Constitution.Com).\n"
    "Contains the operative provisions; the source cover page and table of\n"
    "contents have been removed. Reference text only -- not legal advice.\n"
)
OUT.write_text(header + "\n\n" + clean, encoding="utf-8")

print(f"Wrote {OUT}")
print(f"  {len(clean):,} chars, {len(paras):,} paragraphs")
print("\n--- first 5 paragraphs ---\n")
for p in paras[:5]:
    print(repr(p[:160]))
