"""Extracts the text of the proposal presentation, slide by slide, into presentation_text.txt."""
from pathlib import Path

from pypdf import PdfReader

here = Path(__file__).resolve().parent
reader = PdfReader(here / "IIT Group 16 Research Proposal Presentation.pdf")

with open(here / "presentation_text.txt", "w", encoding="utf-8") as out:
    for i, page in enumerate(reader.pages, start=1):
        out.write(f"\n===== SLIDE {i} =====\n")
        out.write((page.extract_text() or "").strip() + "\n")

print(f"Done: {len(reader.pages)} slides -> presentation_text.txt")
