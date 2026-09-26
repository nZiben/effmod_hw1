"""Generate the legible notebook-style derivation PDF delivered with HW1."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


WIDTH, HEIGHT = A4
MARGIN_X = 42
TOP = HEIGHT - 42
LINE_HEIGHT = 19


def choose_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/segoepr.ttf"),
        Path("C:/Windows/Fonts/comic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("NotebookHand", str(path)))
            return "NotebookHand"
    return "Helvetica"


def page_background(pdf: canvas.Canvas, page_number: int, title: str, font: str) -> float:
    pdf.setFillColor(HexColor("#fffdf6"))
    pdf.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    pdf.setStrokeColor(HexColor("#d9e8f5"))
    for y in range(45, int(HEIGHT - 35), LINE_HEIGHT):
        pdf.line(28, y, WIDTH - 28, y)
    pdf.setStrokeColor(HexColor("#efb1b1"))
    pdf.line(36, 28, 36, HEIGHT - 28)
    pdf.setFillColor(HexColor("#15304a"))
    pdf.setFont(font, 17)
    pdf.drawString(MARGIN_X, TOP, title)
    pdf.setFont(font, 8)
    pdf.drawRightString(WIDTH - 35, 24, f"HW1 derivations  |  page {page_number}/4")
    return TOP - 30


def write_lines(
    pdf: canvas.Canvas,
    y: float,
    lines: list[str],
    font: str,
    size: float = 10.5,
    gap: float = LINE_HEIGHT,
    color: Color = HexColor("#17324d"),
) -> float:
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in lines:
        pdf.drawString(MARGIN_X, y, line)
        y -= gap
    return y


def generate(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    font = choose_font()
    pdf = canvas.Canvas(str(destination), pagesize=A4, pageCompression=1)
    pdf.setTitle("Homework 1 - analytical performance model derivations")
    pdf.setAuthor("ITMO Efficient Models course - reproducible solution")

    y = page_background(pdf, 1, "1. Network shapes and FLOPs", font)
    y = write_lines(
        pdf,
        y,
        [
            "Assumptions: FP32 inference, eval(), B images of shape 3 x S x S, S divisible by 16.",
            "Convolution bias = false; Linear bias = true. One multiply-accumulate = 2 FLOPs.",
            "FLOPs count Conv/Linear arithmetic only (not ReLU, pooling, comparisons or indexing).",
            "",
            "Layer                         output shape             FLOPs",
            "Conv 7x7, 3 -> 32, s2         B x 32 x S/2 x S/2     2352 B S^2",
            "MaxPool 3x3, s2               B x 32 x S/4 x S/4     excluded",
            "Conv 5x5, 32 -> 64            B x 64 x S/4 x S/4     6400 B S^2",
            "Conv 3x3, 64 -> 128, s2       B x 128 x S/8 x S/8    2304 B S^2",
            "Conv 1x1, 128 -> 256          B x 256 x S/8 x S/8    1024 B S^2",
            "Conv 3x3, 256 -> 256, s2      B x 256 x S/16 x S/16  4608 B S^2",
            "Conv 1x1, 256 -> 512          B x 512 x S/16 x S/16  1024 B S^2",
            "Global average pool           B x 512                  excluded",
            "Linear 512 -> 256             B x 256                 262144 B",
            "Linear 256 -> 100             B x 100                  51200 B",
            "",
            "Example: Conv7 FLOPs = 2 * B*(S/2)^2 * 32*(3*7*7) = 2352 B S^2.",
            "Sum of convolution coefficients = 17712; linear term = 313344 B.",
            "",
            "FLOPs(S,B) = B * (17712 S^2 + 313344).",
        ],
        font,
        size=9.4,
        gap=18,
    )
    pdf.showPage()

    y = page_background(pdf, 2, "2. Peak allocated-memory equation", font)
    y = write_lines(
        pdf,
        y,
        [
            "Convention: peak torch.cuda.max_memory_allocated() during one inference-mode forward.",
            "The equation models live FP32 tensors. It excludes the CUDA context, reserved-but-free",
            "cache blocks, allocator rounding and implementation-dependent cuDNN temporary workspace.",
            "",
            "Persistent parameter elements:",
            "  convolutions = 4704+51200+73728+32768+589824+131072 = 883296",
            "  linear weights/biases = 131072+256+25600+100 = 157028",
            "  total = 1040324 elements = 4161296 bytes.",
            "",
            "Because ReLU is in-place, a layer peak is input + output (parameters stay live):",
            "Operation                    input coeff.   output coeff.   live coeff. (x B S^2)",
            "Conv7                         3              8               11   <-- maximum",
            "MaxPool                       8              2               10",
            "Conv5                         2              4                6",
            "Conv3, 64 -> 128              4              2                6",
            "Conv1, 128 -> 256             2              4                6",
            "Conv3, 256 -> 256             4              1                5",
            "Conv1, 256 -> 512             1              2                3",
            "",
            "First-convolution live activation bytes = 4*(3+8) B S^2 = 44 B S^2.",
            "",
            "Memory(S,B) = 4161296 + 44 B S^2 bytes.",
            "",
            "A point is analytically predicted OOM when this value exceeds physical GPU memory.",
            "Measured OOM can happen earlier when cuDNN workspaces or other processes need memory.",
        ],
        font,
        size=9.1,
        gap=18,
    )
    pdf.showPage()

    y = page_background(pdf, 3, "3. Bytes moved and latency roofline", font)
    y = write_lines(
        pdf,
        y,
        [
            "Compulsory-traffic assumption: each logical op reads its input/weights once and writes",
            "its output once. An in-place ReLU reads and writes the same activation. Cache effects,",
            "re-reads, write allocation and cuDNN workspace traffic are not counted.",
            "",
            "Activation-traffic coefficients (elements):",
            "  Conv7 + ReLU + MaxPool                         = 37 B S^2",
            "  Conv5 + ReLU                                   = 14 B S^2",
            "  Conv3(64->128) + ReLU                          = 10 B S^2",
            "  Conv1(128->256) + ReLU                         = 14 B S^2",
            "  Conv3(256->256) + ReLU                          = 7 B S^2",
            "  Conv1(256->512) + ReLU                          = 7 B S^2",
            "  Global average-pool input                       = 2 B S^2",
            "  GAP/linear/head vectors                         = 2148 B elements",
            "  all weights and biases                          = 1040324 elements",
            "",
            "Q(S,B) = 4 * (91 B S^2 + 2148 B + 1040324) bytes.",
            "",
            "Logical launches N = 6 Conv + 7 ReLU + MaxPool + GAP + 2 Linear = 17.",
            "Let L = seconds/logical launch, P = effective FLOP/s, W = effective byte/s.",
            "",
            "Latency(S,B,theta) = 17 L + max(FLOPs(S,B)/P, Q(S,B)/W).",
            "",
            "Interpretation:",
            "  launch-bound: 17L is the largest component (small work);",
            "  memory-bound: Q/W is largest (low arithmetic intensity);",
            "  compute-bound: FLOPs/P is largest (weight traffic is amortized at large B*S^2).",
            "L, P and W are fitted only on the 7 x 9 base grid; added random coordinates are held out.",
        ],
        font,
        size=9.1,
        gap=18,
    )
    pdf.showPage()

    y = page_background(pdf, 4, "4. Whole-GPU energy and validation", font)
    y = write_lines(
        pdf,
        y,
        [
            "Energy convention: joules for one forward pass on the whole board; idle energy is NOT",
            "subtracted. A repeated forward loop is measured and its board energy is divided by the",
            "number of passes. NVML's total-energy counter is preferred; sampled board power is fallback.",
            "",
            "Let t_hat be the latency equation, P_idle an effective background-power coefficient,",
            "e_F joules/FLOP and e_Q joules/byte. All coefficients are constrained non-negative.",
            "",
            "Energy(S,B,theta_E) = P_idle*t_hat + e_F*FLOPs(S,B) + e_Q*Q(S,B).",
            "",
            "Calibration and test protocol:",
            "  - flags: cudnn.benchmark=false; cudnn.allow_tf32=false; matmul.allow_tf32=false",
            "  - model.eval(), torch.inference_mode(), FP32, random CUDA tensors",
            "  - latency: CUDA events; report median over adaptive repeats",
            "  - memory: reset peak statistics immediately before the measured forward",
            "  - base 7 sizes x 9 batches: fit theta (subject to any measured OOM)",
            "  - random S={48,112,352,400}, B={95,98,167}: unseen validation coordinates",
            "  - every point with either random coordinate is validation (69 of 132 total points)",
            "",
            "Expected model breaks:",
            "  - logical launch count is not the true number of GPU kernels launched by cuDNN",
            "  - cuDNN can switch algorithms at shape boundaries, causing latency discontinuities",
            "  - Q is compulsory traffic, not measured DRAM traffic; cache reuse changes effective Q",
            "  - power sensors are temporally filtered; repeated loops approximate per-pass energy",
            "  - concurrent display/desktop work affects whole-board power and available memory",
            "  - the hard roofline max omits overlap and transition curvature.",
            "",
            "Therefore held-out error and residual patterns matter more than training fit alone.",
        ],
        font,
        size=9.15,
        gap=18,
    )
    pdf.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="hw1_handwritten.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(Path(args.output))
    print(f"Wrote {args.output}")
