"""PDF parsing wrapper. Delegates to MinerU (`magic-pdf`) when available.

We deliberately keep this thin and CLI-based: `magic-pdf` is a heavy install with
torch + paddlepaddle, and most users will run it as a one-shot data prep step,
not inside the live app. The output we care about is just `<stem>/auto/<stem>.md`.

If MinerU isn't installed, this falls back to a plain PyMuPDF / pypdf text dump —
useful for smoke-testing the pipeline, NOT for production quality.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def parse_pdf(pdf_path: Path, out_dir: Path, *, force: bool = False) -> Path:
    """Returns the path to the produced markdown file."""
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    if shutil.which("magic-pdf"):
        return _parse_with_mineru(pdf_path, out_dir, stem, force=force)

    print("[parsing] magic-pdf not found — falling back to plain text extraction.")
    print("[parsing] Install MinerU for production quality:")
    print("[parsing]   pip install -U 'magic-pdf[full]' --extra-index-url https://wheels.myhloli.com")
    return _parse_plain(pdf_path, out_dir, stem)


def _parse_with_mineru(pdf_path: Path, out_dir: Path, stem: str, *, force: bool) -> Path:
    expected = out_dir / stem / "auto" / f"{stem}.md"
    if expected.exists() and not force:
        print(f"[parsing] {expected} exists; skipping (use --force to re-parse).")
        return expected

    cmd = [
        "magic-pdf",
        "pdf-command",
        "--pdf", str(pdf_path),
        "--output-dir", str(out_dir),
        "--method", "auto",
    ]
    print(f"[parsing] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if not expected.exists():
        # Some versions output flat; try to locate the .md
        candidates = list((out_dir / stem).rglob("*.md"))
        if candidates:
            return candidates[0]
        raise RuntimeError(f"MinerU finished but no .md found under {out_dir / stem}")
    return expected


def _parse_plain(pdf_path: Path, out_dir: Path, stem: str) -> Path:
    """Fallback: just dump text. Formulas will be garbage — for smoke test only."""
    try:
        import pypdf
    except ImportError:
        raise RuntimeError(
            "Plain fallback needs pypdf. `pip install pypdf` or install magic-pdf."
        )
    reader = pypdf.PdfReader(str(pdf_path))
    out_md = out_dir / f"{stem}.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write(f"# {stem}\n\n")
        for i, page in enumerate(reader.pages):
            f.write(f"\n\n<!-- page {i + 1} -->\n\n")
            f.write(page.extract_text() or "")
    return out_md
