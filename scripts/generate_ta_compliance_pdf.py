#!/usr/bin/env python3
"""
Generate a professional PDF report containing the complete 5 TA Requirements Compliance Table.
Pure-Python implementation with zero third-party dependencies.
"""

from __future__ import annotations
import sys
import zlib
from pathlib import Path

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

def string_width(text: str, size: float = 9.0, bold: bool = False) -> float:
    factor = 1.08 if bold else 1.0
    tot = sum(HELVETICA_WIDTHS.get(c, 556) for c in text)
    return (tot / 1000.0) * size * factor

def wrap_text(text: str, max_width: float, size: float = 9.0, bold: bool = False) -> list[str]:
    lines = []
    paragraphs = text.split("\n")
    for para in paragraphs:
        if not para:
            lines.append("")
            continue
        words = para.split(" ")
        cur_line: list[str] = []
        for w in words:
            test_line = " ".join(cur_line + [w]) if cur_line else w
            if string_width(test_line, size, bold) <= max_width:
                cur_line.append(w)
            else:
                if cur_line:
                    lines.append(" ".join(cur_line))
                    cur_line = [w]
                else:
                    lines.append(w)
                    cur_line = []
        if cur_line:
            lines.append(" ".join(cur_line))
    return lines

def escape_pdf(text: str) -> str:
    text = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = text.replace('—', ' - ').replace('–', '-').replace('…', '...')
    text = text.replace('≈', '~=').replace('≤', '<=').replace('≥', '>=')
    text = text.replace('Δ', 'Delta').replace('•', '*')
    return text

class PageContent:
    def __init__(self, page_num: int):
        self.page_num = page_num
        self.stream: list[str] = []

    def add_op(self, op: str):
        self.stream.append(op)

    def get_data(self, total_pages: int) -> bytes:
        footer_stream = []
        # Running header
        footer_stream.append("q 0.5 0.5 0.5 rg")
        footer_stream.append("0.4 w 36 570 m 756 570 l S")
        footer_stream.append("BT /F2 8 Tf 36 575 Td (KV-Cache Scaling: TA Requirements Verification & Compliance Matrix) Tj ET")
        footer_stream.append("Q")

        # Running footer
        footer_stream.append("q 0.5 0.5 0.5 rg")
        footer_stream.append("0.4 w 36 35 m 756 35 l S")
        footer_stream.append("BT /F1 8 Tf 36 24 Td (NITK Surathkal | Dept of CSE | Devansh Sharma & Veena S | Qwen2.5-1.5B on T4) Tj ET")
        pg_str = f"Page {self.page_num} of {total_pages}"
        pg_w = string_width(pg_str, 8, False)
        footer_stream.append(f"BT /F1 8 Tf {756 - pg_w:.1f} 24 Td ({pg_str}) Tj ET")
        footer_stream.append("Q")

        full = "\n".join(self.stream + footer_stream)
        return full.encode('latin-1', errors='replace')

class TAPDF:
    def __init__(self):
        self.pages: list[PageContent] = []
        self.current_page: PageContent | None = None
        # Landscape Letter: 792 x 612
        self.page_width = 792.0
        self.page_height = 612.0
        self.left = 36.0
        self.right = 756.0
        self.content_width = 720.0
        self.top = 550.0
        self.bottom = 45.0
        self.y = self.top
        self.new_page()

    def new_page(self):
        p = PageContent(len(self.pages) + 1)
        self.pages.append(p)
        self.current_page = p
        self.y = self.top

    def ensure_space(self, h: float):
        if self.y - h < self.bottom:
            self.new_page()

    def add_title_banner(self):
        self.ensure_space(65)
        # Navy blue banner background
        self.current_page.add_op(f"q 0.10 0.22 0.38 rg 36 {self.y - 50:.1f} 720 50 re f Q")
        self.current_page.add_op(f"q 0.85 0.45 0.15 rg 36 {self.y - 52:.1f} 720 2 re f Q")

        # Title text
        self.current_page.add_op("q 1.0 1.0 1.0 rg")
        self.current_page.add_op(f"BT /F2 14 Tf 48 {self.y - 22:.1f} Td (TA Requirements Verification & Compliance Matrix: All 5 Deliverables) Tj ET")
        self.current_page.add_op("Q")

        # Subtitle
        self.current_page.add_op("q 0.85 0.90 0.96 rg")
        sub = "Project: KV-Cache Scaling in Large Language Models (Qwen2.5-1.5B on NVIDIA T4) | Authors: Devansh Sharma & Veena S (NITK)"
        self.current_page.add_op(f"BT /F1 8.5 Tf 48 {self.y - 38:.1f} Td ({escape_pdf(sub)}) Tj ET")
        self.current_page.add_op("Q")

        self.y -= 62

    def add_table(self, headers: list[str], rows: list[list[str]], col_widths: list[float]):
        font_size = 7.5
        line_h = 9.8
        pad_y = 4.0

        hdr_lines = [wrap_text(h, col_widths[i] - 8, font_size, True) for i, h in enumerate(headers)]
        hdr_h = max(len(l) for l in hdr_lines) * line_h + 2 * pad_y

        def draw_hdr():
            self.ensure_space(hdr_h + 10)
            self.current_page.add_op(f"q 0.15 0.30 0.50 rg 36 {self.y - hdr_h:.1f} 720 {hdr_h:.1f} re f Q")
            self.current_page.add_op(f"q 0.08 0.18 0.32 RG 0.5 w 36 {self.y - hdr_h:.1f} 720 {hdr_h:.1f} re S Q")
            x = 36.0
            for i, h_lines in enumerate(hdr_lines):
                cur_y = self.y - pad_y - font_size
                for l in h_lines:
                    self.current_page.add_op("q 1.0 1.0 1.0 rg")
                    self.current_page.add_op(f"BT /F2 {font_size} Tf {x + 4:.1f} {cur_y:.1f} Td ({escape_pdf(l)}) Tj ET")
                    self.current_page.add_op("Q")
                    cur_y -= line_h
                x += col_widths[i]
            self.y -= hdr_h

        draw_hdr()

        for r_idx, row in enumerate(rows):
            cell_lines = [wrap_text(str(cell), col_widths[i] - 8, font_size, False) for i, cell in enumerate(row)]
            row_h = max(len(l) for l in cell_lines) * line_h + 2 * pad_y

            if self.y - row_h < self.bottom:
                self.new_page()
                draw_hdr()

            # Row background
            bg = "0.96 0.97 0.99" if (r_idx % 2 == 1) else "1.0 1.0 1.0"
            self.current_page.add_op(f"q {bg} rg 36 {self.y - row_h:.1f} 720 {row_h:.1f} re f Q")
            self.current_page.add_op(f"q 0.80 0.84 0.88 RG 0.4 w 36 {self.y - row_h:.1f} 720 {row_h:.1f} re S Q")

            # Vertical lines
            x_div = 36.0
            for w in col_widths[:-1]:
                x_div += w
                self.current_page.add_op(f"q 0.80 0.84 0.88 RG 0.4 w {x_div:.1f} {self.y - row_h:.1f} m {x_div:.1f} {self.y:.1f} l S Q")

            # Cell text
            x = 36.0
            for i, c_lines in enumerate(cell_lines):
                cur_y = self.y - pad_y - font_size
                for l_str in c_lines:
                    bold = (i == 0 or (i == 2 and "SATISFIED" in l_str))
                    font = "/F2" if bold else "/F1"
                    if "SATISFIED" in l_str:
                        self.current_page.add_op("q 0.05 0.50 0.20 rg")
                    elif i == 0:
                        self.current_page.add_op("q 0.10 0.20 0.35 rg")
                    else:
                        self.current_page.add_op("q 0.18 0.22 0.28 rg")

                    self.current_page.add_op(f"BT {font} {font_size} Tf {x + 4:.1f} {cur_y:.1f} Td ({escape_pdf(l_str)}) Tj ET")
                    self.current_page.add_op("Q")
                    cur_y -= line_h
                x += col_widths[i]

            self.y -= row_h

    def build_pdf(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        total = len(self.pages)
        pdf_objs = {}
        pdf_objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
        kids = " ".join(f"{7 + 2*i} 0 R" for i in range(total))
        pdf_objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {total} >>".encode('ascii')
        pdf_objs[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        pdf_objs[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
        pdf_objs[5] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"

        for i, page in enumerate(self.pages):
            cid = 6 + 2 * i
            pid = 7 + 2 * i
            raw = page.get_data(total)
            comp = zlib.compress(raw)
            pdf_objs[cid] = f"<< /Length {len(comp)} /Filter /FlateDecode >>\nstream\n".encode('ascii') + comp + b"\nendstream"
            pdf_objs[pid] = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 792 612] "
                f"/Contents {cid} 0 R "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> >>"
            ).encode('ascii')

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = {}
        max_id = 5 + 2 * total
        for oid in range(1, max_id + 1):
            offsets[oid] = len(out)
            out.extend(f"{oid} 0 obj\n".encode('ascii') + pdf_objs[oid] + b"\nendobj\n")

        x_off = len(out)
        out.extend(f"xref\n0 {max_id + 1}\n0000000000 65535 f \n".encode('ascii'))
        for oid in range(1, max_id + 1):
            out.extend(f"{offsets[oid]:010d} 00000 n \n".encode('ascii'))
        out.extend(f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{x_off}\n%%EOF\n".encode('ascii'))
        path.write_bytes(out)
        print(f"[OK] Generated TA Compliance PDF -> {path}")

def generate_ta_pdf():
    doc = TAPDF()
    doc.add_title_banner()

    headers = [
        "Req #",
        "TA Requirement Description",
        "Compliance Status",
        "Exact Experimental Results & Key Findings",
        "Repository File Path & Line Citations"
    ]

    rows = [
        [
            "Item 1",
            "Paired Bisection Down to <= 512 Tokens\n(Default vs. Expandable)",
            "SATISFIED\n(Verified)",
            "* Default Bracket: [118,784, 119,296] (Gap = 512 tokens)\n"
            "* Expandable Bracket: [127,488, 128,000] (Gap = 512 tokens)\n"
            "* Allocator Capacity Gain: Delta = +8,704 tokens (+7.33% usable context gain)\n"
            "* Upper OOM Autopsy (L=131,072): Proven to be a 34.36 GB framework mask cliff in HuggingFace SDPA, not physical KV saturation",
            "artifacts/bisect-final-complete/audit/bisection_summary.json (Lines 1-15)\n"
            "artifacts/bisect-final-complete/state/probes_default.jsonl\n"
            "artifacts/bisect-final-complete/state/probes_expandable.jsonl\n"
            "docs/BISECTION_DATA_AUDIT.md\n"
            "docs/TA_RESPONSE.md (Lines 10-35)"
        ],
        [
            "Item 2",
            "Discrepancy Table with Explicit 'Unaccounted' Row at L=114,688",
            "SATISFIED\n(Verified)",
            "* Predicted Tensor Peak: 15,359,962,112 B (14.65 GiB)\n"
            "* Measured Peak Allocated: 14,020,668,928 B (13.06 GiB)\n"
            "* Signed Gap (Measured - Predicted): -1,339,293,184 B (-1,277.25 MiB)\n"
            "* Unaccounted Post-Load Storage: 0 Bytes (0.00 MiB)\n"
            "* Unaccounted Post-Forward Growth: 0 Bytes (0.00 MiB)\n"
            "* Arithmetic Closure Error: 0 Bytes (Exact zero-residual closure)",
            "docs/DISCREPANCY_114688.md (Lines 1-20)\n"
            "docs/DISCREPANCY_114688.csv (Full breakdown)\n"
            "docs/TA_RESPONSE.md (Lines 37-96)\n"
            "README.md (Section: Discrepancy Accounting at L=114,688)"
        ],
        [
            "Item 3",
            "Methods Section with Exact File & Line Citations",
            "SATISFIED\n(Verified)",
            "* KV Extraction: src/kv_cache_scaling/probe.py:344-346 (DynamicCache tensors)\n"
            "* Peak Counter Reset: probe.py:347 (torch.cuda.reset_peak_memory_stats)\n"
            "* Logits Truncation: probe.py:317 (logits_to_keep=1 avoids 151,936 vocab tensor)\n"
            "* Attention Kernel: probe.py:295-301 (forces PyTorch memory-efficient SDPA)\n"
            "* Allocator Flag: processes.py:12-18 (PYTORCH_CUDA_ALLOC_CONF injected)\n"
            "* Pre-Registration: prereg/formal-2026-10-18-t4/receipt.json (commit 7baff5121)",
            "docs/METHODS.md (Lines 1-50)\n"
            "docs/TA_RESPONSE.md (Lines 98-155)\n"
            "src/kv_cache_scaling/probe.py (Lines 285-388)\n"
            "src/kv_cache_scaling/processes.py (Lines 12-18)\n"
            "README.md (Section: Codebase Architecture & File Citations)"
        ],
        [
            "Item 4",
            "Separating Committed Prediction (Table A) vs. Fitted Model (Table B)",
            "SATISFIED\n(Verified)",
            "* Table A (Committed Prediction L_pred = 111,232 tokens before GPU run):\n"
            "   - Error vs. L=114,688 lower bound: -3.01% (Safe / Conservative)\n"
            "   - Error vs. L=131,071 upper bound: -15.14% (Safe / Conservative)\n"
            "   - Error vs. Default Bisection (118,784): -6.36%\n"
            "* Table B (Post-Hoc Fitted Peak Model, slope = 95,236.95 B/tok):\n"
            "   - Analytical crossing: 130,509 tokens\n"
            "   - Held-out error at L=114,688: -0.0104% (Default), 0.0000% (Expandable)",
            "docs/PREDICTION_ERROR_AT_CURRENT_BRACKET.md (Lines 1-40)\n"
            "docs/TA_RESPONSE.md (Lines 157-220)\n"
            "README.md (Section: Table A vs. Table B)"
        ],
        [
            "Item 5",
            "Published Measured vLLM Benchmark Figure Naming Exact GPU",
            "SATISFIED\n(Verified)",
            "* Target Architecture: Qwen/Qwen3-8B BF16 (36 layers, 8 KV heads, dim 128)\n"
            "* Inference Engine: vLLM v0.30.0 (gpu_memory_utilization = 0.92, max_model_len 8192)\n"
            "* Hardware: NVIDIA A10G 24GB PCIe Gen4 (AWS EC2 g5.xlarge, not SXM)\n"
            "* Published Measured KV Pool: 3.16 GiB (Startup line: 23,040 GPU KV tokens)\n"
            "* Analytical Formula Prediction: floor(3.16 * 2^30 / 147,456) = 23,010 tokens\n"
            "* Quantified Generalization Error: 23,010 - 23,040 = -30 tokens (-0.1302%)",
            "docs/VLLM_PUBLISHED_FIGURE.md (Lines 1-35)\n"
            "docs/TA_RESPONSE.md (Lines 222-275)\n"
            "README.md (Section: Cross-Hardware & Production Validation on vLLM)"
        ]
    ]

    col_widths = [45.0, 140.0, 75.0, 240.0, 220.0]
    doc.add_table(headers, rows, col_widths)

    repo_root = Path(__file__).resolve().parent.parent
    pdf_out1 = repo_root / "report" / "TA_REQUIREMENTS_COMPLIANCE.pdf"
    pdf_out2 = repo_root / "docs" / "TA_REQUIREMENTS_COMPLIANCE.pdf"
    pdf_out3 = Path("/Users/veenashivakumar/Kv-cache /TA_REQUIREMENTS_COMPLIANCE.pdf")
    pdf_out4 = Path("/Users/veenashivakumar/Kv-cache /KV_Cache_Master_Submission/TA_REQUIREMENTS_COMPLIANCE.pdf")

    doc.build_pdf(pdf_out1)
    pdf_out2.write_bytes(pdf_out1.read_bytes())
    pdf_out3.write_bytes(pdf_out1.read_bytes())
    pdf_out4.write_bytes(pdf_out1.read_bytes())
    print("[OK] Synchronized TA Compliance PDF across all 4 locations.")

if __name__ == "__main__":
    generate_ta_pdf()
