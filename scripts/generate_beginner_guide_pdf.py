#!/usr/bin/env python3
"""
Generate a comprehensive, beginner-friendly PDF guide for KV-Cache Scaling.

Pure-Python implementation with zero third-party dependencies.
Produces a clean, publication-quality PDF 1.4 document with:
- Running headers and footers with "Page X of Y"
- Styled title banner and section headings
- Shaded callout boxes (Analogy, Key Concept, Warning)
- Multi-column tables with text wrapping and auto page breaks
- Monospace ASCII architecture diagrams
- Plain-English explanations of all technical terms, Colab, GPU memory, and code.
"""

from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path

# Standard Type 1 Helvetica approximate character widths (1000 units = 1 em)
HELVETICA_WIDTHS = {
    ' ': 278, '!': 278, '"': 355, '#': 556, '$': 556, '%': 889, '&': 667, "'": 191,
    '(': 333, ')': 333, '*': 389, '+': 584, ',': 278, '-': 333, '.': 278, '/': 278,
    '0': 556, '1': 556, '2': 556, '3': 556, '4': 556, '5': 556, '6': 556, '7': 556,
    '8': 556, '9': 556, ':': 278, ';': 278, '<': 584, '=': 584, '>': 584, '?': 556,
    '@': 1015, 'A': 667, 'B': 667, 'C': 722, 'D': 722, 'E': 667, 'F': 611, 'G': 778,
    'H': 722, 'I': 278, 'J': 500, 'K': 667, 'L': 556, 'M': 833, 'N': 722, 'O': 778,
    'P': 667, 'Q': 778, 'R': 722, 'S': 667, 'T': 611, 'U': 722, 'V': 667, 'W': 944,
    'X': 667, 'Y': 667, 'Z': 611, '[': 278, '\\': 278, ']': 278, '^': 469, '_': 556,
    '`': 333, 'a': 556, 'b': 556, 'c': 500, 'd': 556, 'e': 556, 'f': 278, 'g': 556,
    'h': 556, 'i': 222, 'j': 222, 'k': 500, 'l': 222, 'm': 833, 'n': 556, 'o': 556,
    'p': 556, 'q': 556, 'r': 333, 's': 500, 't': 278, 'u': 556, 'v': 500, 'w': 722,
    'x': 500, 'y': 500, 'z': 500, '{': 334, '|': 260, '}': 334, '~': 584
}

def string_width(text: str, size: float = 10.0, bold: bool = False, mono: bool = False) -> float:
    if mono:
        return len(text) * size * 0.60
    factor = 1.08 if bold else 1.0
    tot = sum(HELVETICA_WIDTHS.get(c, 556) for c in text)
    return (tot / 1000.0) * size * factor

def wrap_text(text: str, max_width: float, size: float = 10.0, bold: bool = False, mono: bool = False) -> list[str]:
    words = text.split(' ')
    lines = []
    cur_line: list[str] = []
    for w in words:
        test_line = ' '.join(cur_line + [w]) if cur_line else w
        if string_width(test_line, size, bold, mono) <= max_width:
            cur_line.append(w)
        else:
            if cur_line:
                lines.append(' '.join(cur_line))
                cur_line = [w]
            else:
                lines.append(w)
                cur_line = []
    if cur_line:
        lines.append(' '.join(cur_line))
    return lines

def escape_pdf(text: str) -> str:
    # Clean text to safe Latin-1 / ASCII
    text = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = text.replace('—', ' - ').replace('–', '-').replace('…', '...')
    text = text.replace('≈', '~=').replace('≤', '<=').replace('≥', '>=')
    text = text.replace('Δ', 'Delta').replace('•', '*')
    return text

class PageContent:
    def __init__(self, page_num: int):
        self.page_num = page_num
        self.stream = []

    def add_op(self, op: str):
        self.stream.append(op)

    def get_data(self, total_pages: int) -> bytes:
        # Add running header and footer if page > 1
        footer_stream = []
        if self.page_num > 1:
            # Running header
            footer_stream.append("q 0.5 0.5 0.5 rg")
            footer_stream.append("0.4 w 54 750 m 558 750 l S")
            footer_stream.append("BT /F1 8 Tf 54 755 Td (KV-Cache Scaling: The Complete Beginner's Guide) Tj ET")
            footer_stream.append("Q")

        # Running footer on all pages
        footer_stream.append("q 0.5 0.5 0.5 rg")
        footer_stream.append("0.4 w 54 45 m 558 45 l S")
        footer_stream.append("BT /F1 8 Tf 54 34 Td (Google Colab & Nvidia Tesla T4 - Qwen2.5-1.5B FP16) Tj ET")
        pg_str = f"Page {self.page_num} of {total_pages}"
        pg_w = string_width(pg_str, 8, False, False)
        footer_stream.append(f"BT /F1 8 Tf {558 - pg_w:.1f} 34 Td ({pg_str}) Tj ET")
        footer_stream.append("Q")

        full_content = "\n".join(self.stream + footer_stream)
        return full_content.encode('latin-1', errors='replace')


class BeginnerGuidePDF:
    def __init__(self):
        self.pages: list[PageContent] = []
        self.current_page: PageContent | None = None
        self.page_width = 612.0
        self.page_height = 792.0
        self.left_margin = 54.0
        self.right_margin = 558.0
        self.content_width = 504.0
        self.top_margin = 735.0
        self.bottom_margin = 58.0
        self.y = self.top_margin
        self.new_page()

    def new_page(self):
        p = PageContent(len(self.pages) + 1)
        self.pages.append(p)
        self.current_page = p
        self.y = self.top_margin

    def ensure_space(self, needed_height: float):
        if self.y - needed_height < self.bottom_margin:
            self.new_page()

    def add_title_banner(self, title: str, subtitle: str, meta_items: list[str]):
        self.ensure_space(110)
        # Background shaded card
        self.current_page.add_op(f"q 0.94 0.96 0.98 rg 54 {self.y - 100:.1f} 504 100 re f Q")
        # Navy accent border top
        self.current_page.add_op(f"q 0.10 0.21 0.36 rg 54 {self.y - 4:.1f} 504 4 re f Q")
        # Outer border
        self.current_page.add_op(f"q 0.85 0.88 0.92 RG 0.5 w 54 {self.y - 100:.1f} 504 100 re S Q")

        # Title
        self.current_page.add_op("q 0.10 0.21 0.36 rg")
        self.current_page.add_op(f"BT /F2 17 Tf 70 {self.y - 28:.1f} Td ({escape_pdf(title)}) Tj ET")
        self.current_page.add_op("Q")

        # Subtitle
        self.current_page.add_op("q 0.17 0.42 0.69 rg")
        self.current_page.add_op(f"BT /F2 10.5 Tf 70 {self.y - 48:.1f} Td ({escape_pdf(subtitle)}) Tj ET")
        self.current_page.add_op("Q")

        # Metadata pill items
        meta_str = " | ".join(meta_items)
        self.current_page.add_op("q 0.35 0.40 0.48 rg")
        self.current_page.add_op(f"BT /F1 8.5 Tf 70 {self.y - 75:.1f} Td ({escape_pdf(meta_str)}) Tj ET")
        self.current_page.add_op("Q")

        self.y -= 118

    def add_heading1(self, text: str):
        self.ensure_space(38)
        self.y -= 10
        # Title
        self.current_page.add_op("q 0.10 0.21 0.36 rg")
        self.current_page.add_op(f"BT /F2 13 Tf 54 {self.y:.1f} Td ({escape_pdf(text)}) Tj ET")
        self.current_page.add_op("Q")
        # Subtle divider
        self.current_page.add_op(f"q 0.17 0.42 0.69 RG 1.0 w 54 {self.y - 4:.1f} 558 {self.y - 4:.1f} m l S Q")
        self.y -= 18

    def add_heading2(self, text: str):
        self.ensure_space(28)
        self.y -= 8
        self.current_page.add_op("q 0.17 0.42 0.69 rg")
        self.current_page.add_op(f"BT /F2 10.5 Tf 54 {self.y:.1f} Td ({escape_pdf(text)}) Tj ET")
        self.current_page.add_op("Q")
        self.y -= 14

    def add_paragraph(self, text: str, bold_prefix: str = ""):
        full_text = f"{bold_prefix} {text}".strip() if bold_prefix else text
        lines = wrap_text(full_text, self.content_width, 9.0, False, False)
        line_h = 12.5
        needed = len(lines) * line_h + 5
        self.ensure_space(needed)

        for line in lines:
            self.current_page.add_op("q 0.18 0.22 0.28 rg")
            self.current_page.add_op(f"BT /F1 9.0 Tf 54 {self.y:.1f} Td ({escape_pdf(line)}) Tj ET")
            self.current_page.add_op("Q")
            self.y -= line_h
        self.y -= 4

    def add_bullet(self, title: str, text: str):
        full_text = f"{title}: {text}" if title else text
        lines = wrap_text(full_text, self.content_width - 16, 9.0, False, False)
        line_h = 12.0
        needed = len(lines) * line_h + 3
        self.ensure_space(needed)

        # Draw bullet dot
        dot_y = self.y + 3.0
        self.current_page.add_op(f"q 0.17 0.42 0.69 rg 57 {dot_y:.1f} 3.5 3.5 re f Q")

        for i, line in enumerate(lines):
            self.current_page.add_op("q 0.18 0.22 0.28 rg")
            font = "/F2" if (i == 0 and title and not text.startswith(line)) else "/F1"
            self.current_page.add_op(f"BT /F1 9.0 Tf 68 {self.y:.1f} Td ({escape_pdf(line)}) Tj ET")
            self.current_page.add_op("Q")
            self.y -= line_h
        self.y -= 2

    def add_callout(self, title: str, text: str, kind: str = "analogy"):
        lines = wrap_text(text, self.content_width - 24, 8.5, False, False)
        line_h = 11.5
        box_h = 24 + len(lines) * line_h + 8
        self.ensure_space(box_h + 6)

        # Colors based on kind
        if kind == "analogy":
            bg = "0.94 0.97 1.0"      # Light blue
            border = "0.20 0.50 0.85"  # Blue accent
            hdr_color = "0.10 0.30 0.60"
        elif kind == "warning":
            bg = "1.0 0.96 0.94"      # Light orange
            border = "0.85 0.35 0.15"  # Orange accent
            hdr_color = "0.70 0.20 0.05"
        else: # "concept"
            bg = "0.94 0.98 0.95"      # Light green
            border = "0.20 0.65 0.35"  # Green accent
            hdr_color = "0.10 0.45 0.20"

        # Background fill
        self.current_page.add_op(f"q {bg} rg 54 {self.y - box_h:.1f} 504 {box_h:.1f} re f Q")
        # Left border bar
        self.current_page.add_op(f"q {border} rg 54 {self.y - box_h:.1f} 3.5 {box_h:.1f} re f Q")
        # Outline
        self.current_page.add_op(f"q 0.85 0.88 0.92 RG 0.4 w 54 {self.y - box_h:.1f} 504 {box_h:.1f} re S Q")

        # Header title
        self.current_page.add_op(f"q {hdr_color} rg")
        self.current_page.add_op(f"BT /F2 9.5 Tf 68 {self.y - 15:.1f} Td ({escape_pdf(title)}) Tj ET")
        self.current_page.add_op("Q")

        # Body text
        cur_text_y = self.y - 28
        for line in lines:
            self.current_page.add_op("q 0.20 0.24 0.30 rg")
            self.current_page.add_op(f"BT /F1 8.5 Tf 68 {cur_text_y:.1f} Td ({escape_pdf(line)}) Tj ET")
            self.current_page.add_op("Q")
            cur_text_y -= line_h

        self.y -= (box_h + 8)

    def add_table(self, headers: list[str], rows: list[list[str]], col_widths: list[float]):
        row_pad = 4.0
        font_size = 8.0
        line_h = 10.5

        # Calculate heights for header
        hdr_lines = [wrap_text(h, col_widths[i] - 10, font_size, True, False) for i, h in enumerate(headers)]
        hdr_h = max(len(l) for l in hdr_lines) * line_h + 2 * row_pad

        def draw_header():
            self.ensure_space(hdr_h + 10)
            # Header background
            self.current_page.add_op(f"q 0.17 0.42 0.69 rg 54 {self.y - hdr_h:.1f} 504 {hdr_h:.1f} re f Q")
            x = 54.0
            for i, h_lines in enumerate(hdr_lines):
                cur_y = self.y - row_pad - font_size
                for l in h_lines:
                    self.current_page.add_op("q 1.0 1.0 1.0 rg")
                    self.current_page.add_op(f"BT /F2 {font_size} Tf {x + 5:.1f} {cur_y:.1f} Td ({escape_pdf(l)}) Tj ET")
                    self.current_page.add_op("Q")
                    cur_y -= line_h
                x += col_widths[i]
            # Outer header border
            self.current_page.add_op(f"q 0.10 0.25 0.45 RG 0.5 w 54 {self.y - hdr_h:.1f} 504 {hdr_h:.1f} re S Q")
            self.y -= hdr_h

        draw_header()

        for row_idx, row in enumerate(rows):
            cell_lines = [wrap_text(str(cell), col_widths[i] - 10, font_size, False, False) for i, cell in enumerate(row)]
            row_h = max(len(l) for l in cell_lines) * line_h + 2 * row_pad

            if self.y - row_h < self.bottom_margin:
                self.new_page()
                draw_header()

            # Row background
            bg = "0.97 0.98 0.99" if (row_idx % 2 == 1) else "1.0 1.0 1.0"
            self.current_page.add_op(f"q {bg} rg 54 {self.y - row_h:.1f} 504 {row_h:.1f} re f Q")
            # Cell border
            self.current_page.add_op(f"q 0.85 0.88 0.92 RG 0.4 w 54 {self.y - row_h:.1f} 504 {row_h:.1f} re S Q")

            # Draw vertical column dividers
            x_div = 54.0
            for w in col_widths[:-1]:
                x_div += w
                self.current_page.add_op(f"q 0.85 0.88 0.92 RG 0.4 w {x_div:.1f} {self.y - row_h:.1f} m {x_div:.1f} {self.y:.1f} l S Q")

            # Text
            x = 54.0
            for i, c_lines in enumerate(cell_lines):
                cur_y = self.y - row_pad - font_size
                for l in c_lines:
                    self.current_page.add_op("q 0.18 0.22 0.28 rg")
                    self.current_page.add_op(f"BT /F1 {font_size} Tf {x + 5:.1f} {cur_y:.1f} Td ({escape_pdf(l)}) Tj ET")
                    self.current_page.add_op("Q")
                    cur_y -= line_h
                x += col_widths[i]

            self.y -= row_h
        self.y -= 6

    def add_code_block(self, lines: list[str]):
        font_size = 7.5
        line_h = 9.5
        box_h = len(lines) * line_h + 10
        self.ensure_space(box_h + 6)

        # Background
        self.current_page.add_op(f"q 0.95 0.96 0.97 rg 54 {self.y - box_h:.1f} 504 {box_h:.1f} re f Q")
        # Border
        self.current_page.add_op(f"q 0.82 0.85 0.88 RG 0.4 w 54 {self.y - box_h:.1f} 504 {box_h:.1f} re S Q")

        cur_y = self.y - 7 - font_size
        for l in lines:
            self.current_page.add_op("q 0.15 0.20 0.30 rg")
            self.current_page.add_op(f"BT /F3 {font_size} Tf 62 {cur_y:.1f} Td ({escape_pdf(l)}) Tj ET")
            self.current_page.add_op("Q")
            cur_y -= line_h

        self.y -= (box_h + 6)

    def build_pdf(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_pages = len(self.pages)

        # Build PDF objects
        # 1: Catalog
        # 2: Pages
        # 3: Font Helvetica
        # 4: Font Helvetica-Bold
        # 5: Font Courier
        # Then for each page:
        # 6 + 2*i: Content stream
        # 7 + 2*i: Page object

        pdf_objects: dict[int, bytes] = {}
        pdf_objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

        kids = " ".join(f"{7 + 2*i} 0 R" for i in range(total_pages))
        pdf_objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {total_pages} >>".encode('ascii')

        pdf_objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
        pdf_objects[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
        pdf_objects[5] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"

        for i, page in enumerate(self.pages):
            content_bytes = page.get_data(total_pages)
            comp = zlib.compress(content_bytes)
            content_obj_id = 6 + 2 * i
            page_obj_id = 7 + 2 * i

            stream_dict = f"<< /Length {len(comp)} /Filter /FlateDecode >>\nstream\n".encode('ascii') + comp + b"\nendstream"
            pdf_objects[content_obj_id] = stream_dict

            page_dict = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> "
                f"/Contents {content_obj_id} 0 R >>"
            ).encode('ascii')
            pdf_objects[page_obj_id] = page_dict

        max_obj_id = 5 + 2 * total_pages

        with open(output_path, 'wb') as f:
            f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
            offsets: dict[int, int] = {}

            for obj_id in range(1, max_obj_id + 1):
                offsets[obj_id] = f.tell()
                f.write(f"{obj_id} 0 obj\n".encode('ascii'))
                f.write(pdf_objects[obj_id])
                f.write(b"\nendobj\n")

            xref_start = f.tell()
            f.write(f"xref\n0 {max_obj_id + 1}\n".encode('ascii'))
            f.write(b"0000000000 65535 f \n")
            for obj_id in range(1, max_obj_id + 1):
                f.write(f"{offsets[obj_id]:010d} 00000 n \n".encode('ascii'))

            f.write(f"trailer\n<< /Size {max_obj_id + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode('ascii'))

        print(f"Successfully generated {total_pages}-page PDF at: {output_path}")


def build_guide():
    pdf = BeginnerGuidePDF()

    # Title Banner
    pdf.add_title_banner(
        title="The Beginner's Guide to KV-Cache Scaling",
        subtitle="Demystifying LLM Memory, Google Colab, and GPU Architecture",
        meta_items=[
            "Project: KV-Cache-Scaling",
            "Model: Qwen2.5-1.5B (FP16)",
            "Hardware: Nvidia Tesla T4 (15.8 GB VRAM)",
            "Auditor: Antigravity AI"
        ]
    )

    # Chapter 1
    pdf.add_heading1("1. The Big Picture: What is This Project About?")
    pdf.add_paragraph(
        "When you chat with an AI model (like ChatGPT, Claude, or Qwen), it must read your prompt and remember "
        "every previous word to produce meaningful answers. But memory is not free. Every single word you feed "
        "into the AI takes up physical high-speed memory inside the graphics chip (the GPU). If the conversation "
        "becomes too long, the computer runs out of memory and crashes with a CUDA Out of Memory (OOM) error."
    )
    pdf.add_callout(
        "The Desk and Index Cards Analogy",
        "Imagine an office desk of fixed size (15.8 GB on a Tesla T4 GPU). The AI model weights are like a heavy "
        "3.09 GB encyclopedia permanently sitting on the desk. As the AI reads words, it writes notes on index cards "
        "(the KV Cache) at a rate of 28,672 bytes per word. At 50,000 words, half the desk is covered. At 114,688 words, "
        "the desk is almost full. When the cards spill over the edge, the computer crashes!",
        "analogy"
    )
    pdf.add_paragraph(
        "The central mission of this project is rigorous scientific engineering: Can we write a pure mathematical "
        "formula predicting the exact token count where the GPU crashes down to the single byte before running code, "
        "prove that real GPU measurements match our formula within 0.01%, and pinpoint the exact breaking point?"
    )

    # Chapter 2
    pdf.add_heading1("2. The Jargon Buster: Every Technical Term Explained")
    pdf.add_paragraph(
        "Here is your plain-English translation table for every technical concept used in this repository:"
    )

    jargon_headers = ["Technical Term", "Plain-English Meaning", "Everyday Analogy"]
    jargon_widths = [110.0, 224.0, 170.0]
    jargon_rows = [
        ["LLM (Large Language Model)", "A deep learning neural network trained to predict the next word in a sequence (e.g., Qwen2.5-1.5B).", "A super-advanced autocomplete system."],
        ["Token", "A chunk of text roughly 3 to 4 English characters long. The word 'fantastic' is 3 tokens: fan, tas, tic.", "A syllable or word puzzle block."],
        ["Context Length (L)", "The total number of tokens the AI reads and remembers at once (e.g., 512 = 1 page; 114,688 = 300 pages).", "The thickness of the book handed to the AI."],
        ["GPU & VRAM", "Graphics Processing Unit and Video RAM. A specialized chip with 15.8 GB of ultra-fast memory.", "A factory with 2,500 math workers and a large desk."],
        ["FP16 (Half Precision)", "A 16-bit number format taking 2 bytes per value, halving memory usage compared to standard 32-bit floats.", "Writing in fine-point pen to save notebook space."],
        ["KV Cache", "Key-Value Cache. Past attention tables stored in VRAM so the AI doesn't re-calculate past words for every token.", "The index cards the AI keeps on its desk."],
        ["Slope (28,672 B/tok)", "The exact rate memory expands per added token: 28 layers * 2 tensors * 2 heads * 128 dim * 2 bytes.", "The weight of each single index card added."],
        ["CUDA OOM", "Out of Memory crash when the program asks the GPU for 1 more byte than physically available.", "Trying to pour a gallon of water into a pint glass."],
        ["Preregistration", "Formally writing down and Git-locking predictions before running experiments to eliminate hindsight bias.", "Sealing your race prediction in a notarized envelope."],
        ["PyTorch Allocator", "The internal software manager allocating VRAM blocks (Default mode vs modern expandable_segments:True).", "The office organizer stacking boxes on the desk."],
        ["Bisection", "A smart binary search algorithm that cuts the unknown range in half every step to find boundaries fast.", "Playing the Higher-or-Lower guessing game."]
    ]
    pdf.add_table(jargon_headers, jargon_rows, jargon_widths)

    # Chapter 3
    pdf.add_heading1("3. Demystifying Google Colab: How and Why We Use It")
    pdf.add_paragraph(
        "Why Google Colab? Personal laptops (including MacBooks) do not have Nvidia server GPUs. Modern AI libraries "
        "like PyTorch CUDA require Nvidia hardware. Google Colaboratory (Colab) connects your web browser to a free "
        "cloud Linux server in Google's data center equipped with a dedicated Nvidia Tesla T4 GPU (15.8 GB VRAM)."
    )
    pdf.add_bullet("Jupyter Notebooks (.ipynb)", "Interactive documents combining formatted text, visual diagrams, and runnable code.")
    pdf.add_bullet("Notebook Cells", "Stacked blocks. Markdown cells display reading text; Code cells contain Python instructions.")
    pdf.add_bullet("Running a Cell", "Clicking the Play button sends code over the internet to Google's cloud server to execute on the GPU.")
    pdf.add_bullet("How Code Uses the GPU", "Calls to torch.cuda load weights into VRAM and allocate KV-cache memory layer by layer.")

    # Chapter 4
    pdf.add_heading1("4. What Have We Done So Far? (The Project Journey)")
    pdf.add_paragraph(
        "Here is the chronological progression of work in the KV-Cache-Scaling project:"
    )
    pdf.add_bullet("Phase 1 - Mathematical Derivation", "Derived KV slope = 28,672 bytes/token and preregistered 111,232 token capacity.")
    pdf.add_bullet("Phase 2 - 52 GPU Probes", "Ran 52 tests on Colab T4 across lengths 512 to 114,688. Proved 114,688 fits with R^2 = 1.000000.")
    pdf.add_bullet("Part 0 - The Scientific Audit", "Audited claims C1-C9 against Git commits and JSON logs. Proved slope and conservatism pass.")
    pdf.add_bullet("Part 1 - TA Requests 2 to 5", "Produced discrepancy ledger closing to 0 B unaccounted, wrote Methods with code lines, and compared models.")
    pdf.add_bullet("Part 1 - Task 1 (Bisection)", "Created prospective preregistration amendment, isolated worker, and BOUNDARY_COLAB.ipynb.")

    # Chapter 5
    pdf.add_heading1("5. Tour of the Project Folder: Files and Scripts Explained")
    pdf.add_paragraph(
        "Here is a map of the repository files and what each Python script accomplishes:"
    )

    folder_headers = ["Folder / File Path", "Role in Project", "Key Python Functionality"]
    folder_widths = [140.0, 164.0, 200.0]
    folder_rows = [
        ["configs/", "Configuration files", "Defines Qwen2.5-1.5B architecture, FP16 precision, and Tesla T4 specs."],
        ["src/kv_cache_scaling/model_math.py", "Pure Math Library", "Calculates KV slope (28,672 B/tok) and headroom with zero GPU dependencies."],
        ["src/kv_cache_scaling/probe.py", "Memory Probe Engine", "Loads model into VRAM, generates synthetic prompts, and records max_memory_allocated()."],
        ["src/kv_cache_scaling/bisect_worker.py", "Isolated Worker", "Executes a single probe in a fresh subprocess so memory leaks cannot contaminate runs."],
        ["src/kv_cache_scaling/processes.py", "Subprocess Manager", "Launches worker processes safely, capturing exit codes and CUDA OOM exceptions."],
        ["scripts/bisect_boundary.py", "Bisection Coordinator", "Calculates midpoints, narrows brackets, and manages the bisection search loop."],
        ["scripts/create_boundary_notebook.py", "Notebook Generator", "Builds the clean, executable BOUNDARY_COLAB.ipynb notebook automatically."],
        ["notebooks/BOUNDARY_COLAB.ipynb", "Colab Experiment", "The exact 11-cell interactive notebook you run on Google Colab with a T4 GPU."],
        ["prereg/bisect-2026-10-28/", "Preregistration Vault", "Prospective scientific plan freezing OOM budget (<=6 per allocator) before GPU testing."],
        ["docs/TA_RESPONSE.md", "TA Deliverable Report", "Comprehensive 5-item report answering all Teaching Assistant requirements."]
    ]
    pdf.add_table(folder_headers, folder_rows, folder_widths)

    # Chapter 6
    pdf.add_heading1("6. Under the Hood: How the Code Executes on the GPU")
    pdf.add_paragraph(
        "When a probe executes on the Nvidia Tesla T4 GPU, it follows a strict 7-step lifecycle:"
    )

    diagram_lines = [
        "+-------------------------------------------------------------------------------+",
        "| Step 1: Set PYTORCH_CUDA_ALLOC_CONF (Default or expandable_segments:True)    |",
        "| Step 2: Spawn isolated Python worker process with clean GPU memory            |",
        "| Step 3: Load Qwen2.5-1.5B FP16 weights into VRAM (~3.09 GB static base)      |",
        "| Step 4: Allocate synthetic input token sequence of length L                   |",
        "| Step 5: Forward pass: 28 layers store KV tensors in DynamicCache (28,672 B/t) |",
        "| Step 6: Success -> read max_memory_allocated() | OOM -> catch OutOfMemoryError|",
        "| Step 7: Record receipt JSON with timestamp, GPU UUID, and exit cleanly        |",
        "+-------------------------------------------------------------------------------+"
    ]
    pdf.add_code_block(diagram_lines)

    pdf.add_callout(
        "Why Does 131,072 Crash Instantly? (The Attention Mask Cliff)",
        "In Hugging Face transformers, when context length L reaches or exceeds the model sliding_window (131,072), "
        "SDPA attention mask elision is disabled. The library attempts to allocate a dense 2D attention mask of shape "
        "[131072, 131073] in FP16 = 34.36 GB! Since the T4 GPU has only 15.8 GB, it crashes immediately with an OOM. "
        "This makes our bisection testing between 114,688 and 131,072 scientifically essential.",
        "warning"
    )

    # Chapter 7
    pdf.add_heading1("7. Step-by-Step Instructions: Running in Google Colab")
    pdf.add_paragraph(
        "Follow these 6 simple steps to execute the bisection experiment on Google Colab:"
    )
    pdf.add_bullet("Step 1: Locate the Notebook", "Find notebooks/BOUNDARY_COLAB.ipynb in this project directory.")
    pdf.add_bullet("Step 2: Upload to Colab", "Visit colab.research.google.com, click the 'Upload' tab, and drag BOUNDARY_COLAB.ipynb.")
    pdf.add_bullet("Step 3: Enable T4 GPU", "Click Runtime -> Change runtime type -> select 'T4 GPU' -> Save -> Connect.")
    pdf.add_bullet("Step 4: Run Cells Sequentially", "Click the Play button on Cell 1 through Cell 11 in order.")
    pdf.add_bullet("Step 5: Observe Bisection", "Watch Cell 5 (Default allocator) and Cell 7 (Expandable allocator) automatically test midpoints.")
    pdf.add_bullet("Step 6: Download Capsule Zip", "Cell 11 automatically triggers a download of bisect_boundary_t4_YYYYMMDD.zip to your computer.")

    # Chapter 8
    pdf.add_heading1("8. Summary Cheat Sheet")
    cheat_headers = ["Key Question", "Direct Plain-English Answer"]
    cheat_widths = [180.0, 324.0]
    cheat_rows = [
        ["What AI model are we studying?", "Qwen2.5-1.5B in 16-bit half precision (FP16)."],
        ["What hardware are we using?", "Nvidia Tesla T4 GPU in Google Colab (15.8 GB VRAM)."],
        ["What is the memory growth slope?", "Exactly 28,672 bytes per token (R^2 = 1.000000)."],
        ["What was our original prediction?", "111,232 tokens (conservative by ~3% to guarantee safety)."],
        ["What is the highest tested success?", "114,688 tokens (fits comfortably within 0.01% of theory)."],
        ["What is the known failure limit?", "131,072 tokens (hard crash due to 34.36 GB attention mask cliff)."],
        ["What is the goal of bisection?", "Narrow down the boundary between 114,688 and 131,072 to <= 512 tokens."],
        ["What is the Colab notebook to run?", "notebooks/BOUNDARY_COLAB.ipynb"],
        ["What to say when done?", "Download the zip file, put it in artifacts/, and write: 'RESULTS READY'."]
    ]
    pdf.add_table(cheat_headers, cheat_rows, cheat_widths)

    # Output path
    out_file = Path("/Users/veenashivakumar/Kv-cache /KV-Cache-Scaling/docs/PROJECT_EXPLANATION_BEGINNER_GUIDE.pdf")
    pdf.build_pdf(out_file)
    return out_file

if __name__ == "__main__":
    build_guide()
