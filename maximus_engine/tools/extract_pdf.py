import sys
import pymupdf

src, out = sys.argv[1], sys.argv[2]
doc = pymupdf.open(src)
with open(out, "w", encoding="utf-8") as f:
    f.write(f"# pages: {len(doc)}\n")
    for i, page in enumerate(doc, 1):
        f.write(f"\n===== PAGE {i} =====\n")
        f.write(page.get_text())
print(src.split("\\")[-1][:40], "pages:", len(doc))
