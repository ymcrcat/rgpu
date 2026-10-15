"""Render paper.md and figures. Requires reportlab, matplotlib, markdown, pyphen."""
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rgpu-paper-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import markdown
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                               Spacer, Table, TableStyle, Image, Preformatted)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
DATA = json.loads((ROOT / "results.json").read_text())
SOURCE = (ROOT / "paper.md").read_text()
S = DATA["summary"]
assert len(DATA["samples"]) == 35
assert round(S["eager_flush64"]["median_ms"], 3) == 45.421
assert round(S["graph_eager"]["median_ms"], 3) == 2.065
assert S["eager_flush64"]["messages"] == 601
assert S["graph_eager"]["messages"] == 41
assert S["eager_sync_each_call"]["waits"] == 21
assert "40 passed" in (ROOT / "checks.txt").read_text()

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.4, 2.7), layout="constrained")
labels = ["Native CPU", "Eager, flush 1", "Eager, flush 64",
          "Eager, sync/call", "Graph, server eager"]
for i, key in enumerate(S):
    samples = [r["elapsed_ms"] for r in DATA["samples"] if r["case"] == key]
    ax.scatter(samples, [i + (j - 3) * .04 for j in range(7)],
               s=14, color="#819bb2", alpha=.8)
    ax.plot(S[key]["median_ms"], i, "|", color="#0e3655", markersize=16, markeredgewidth=2)
ax.set_yticks(range(5), labels)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Completed execution for 20 calls (ms, log scale)")
ax.grid(axis="x", alpha=.2)
ax.set_title("CPU loopback: seven samples per configuration", fontsize=9)
fig.savefig(FIG / "loopback.png", dpi=300)
fig.savefig(FIG / "loopback.pdf")
plt.close(fig)

# Same-GPU experiment (Section 6.8): native / colocated / WAN on one L40S.
# Each leg is a separate JSON from gpu_benchmark.py; a leg that has not been
# run yet is simply left off the plot rather than failing the render, since
# the assertions above guard only the CPU experiment the manuscript is built on.
GPU_LEGS = [("native", "Native"), ("colocated", "Colocated"), ("wan", "WAN")]
gpu_data = {}
for leg, _ in GPU_LEGS:
    p = ROOT / f"{leg}.json"
    if p.exists() and p.stat().st_size > 0:
        gpu_data[leg] = json.loads(p.read_text())
if gpu_data:
    GPU_ROWS = [("chain/native", "Chain, native"),
                ("chain/eager_flush64", "Chain, eager flush 64"),
                ("chain/eager_sync_each_call", "Chain, eager sync/call"),
                ("chain/graph_eager", "Chain, graph"),
                ("train/native", "Train, native"),
                ("train/eager_flush64", "Train, eager flush 64"),
                ("train/eager_sync_each_step", "Train, eager loss read/step")]
    leg_colors = {"native": "#0e3655", "colocated": "#819bb2", "wan": "#c2703d"}
    fig, ax = plt.subplots(figsize=(4.4, 3.1), layout="constrained")
    labeled = set()
    for i, (key, _) in enumerate(GPU_ROWS):
        for k, (leg, legname) in enumerate(GPU_LEGS):
            d = gpu_data.get(leg)
            if not d or key not in d["summary"]:
                continue
            wl, case = key.split("/")
            samples = [r["elapsed_ms"] for r in d["samples"]
                       if r["workload"] == wl and r["case"] == case]
            off = (k - 1) * .22
            ax.scatter(samples, [i + off] * len(samples), s=12,
                       color=leg_colors[leg], alpha=.75,
                       label=None if leg in labeled else legname)
            labeled.add(leg)
            ax.plot(d["summary"][key]["median_ms"], i + off, "|",
                    color=leg_colors[leg], markersize=13, markeredgewidth=2)
    ax.set_yticks(range(len(GPU_ROWS)), [lab for _, lab in GPU_ROWS])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Completed execution per timed block (ms, log scale)")
    ax.grid(axis="x", alpha=.2)
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_title("One L40S, seven samples per configuration", fontsize=9)
    fig.savefig(FIG / "same_gpu.png", dpi=300)
    fig.savefig(FIG / "same_gpu.pdf")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(4.4, 3.3))
ax.set(xlim=(0, 10), ylim=(0, 10))
ax.axis("off")
for x, title in [(2.5, "PyTorch device path"), (7.5, "CUDA compatibility path")]:
    ax.text(x, 9.7, title, ha="center", va="top", fontsize=9, weight="bold")
boxes = [(0.3, 7.15, "Local application + autograd\nRemoteTensor metadata / IDs"),
         (5.3, 7.15, "Local Linux application\nCUDA / math-library shims"),
         (0.3, 4.1, "Ordered ATen operations\nor registered graph calls"),
         (5.3, 4.1, "Batched CUDA requests\nClient thread / request IDs"),
         (0.3, .8, "Python opserver\nTensor / graph session state\nCUDA, MPS, or CPU backend"),
         (5.3, .8, "C++ CUDA server\nHandle / thread session state\nReal CUDA libraries + GPU")]
for x, y, label in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 4.4, 1.8, boxstyle="round,pad=.08",
                              edgecolor="#627a8d", facecolor="#f0f4f7", linewidth=.8))
    ax.text(x + 2.2, y + .9, label, ha="center", va="center", fontsize=8)
for x in (2.5, 7.5):
    for y1, y2 in [(7.0, 6.0), (3.95, 2.75)]:
        ax.annotate("", (x, y2), (x, y1), arrowprops={"arrowstyle": "->", "color": "#254a66"})
ax.axhline(3.5, color="#8a8a8a", linestyle="--", linewidth=.7)
ax.text(5, 3.58, "network boundary", ha="center", fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
fig.tight_layout(pad=.4)
fig.savefig(FIG / "architecture.png", dpi=300)
fig.savefig(FIG / "architecture.pdf")
plt.close(fig)

width = (612 - 84 - 18) / 2
styles = {
    "body": ParagraphStyle("body", fontName="Times-Roman", fontSize=9.5, leading=11.6,
                           spaceAfter=5, alignment=4, splitLongWords=True,
                           # Justified text in a narrow column needs hyphenation, or a
                           # line ahead of a long word is spread thin. Needs pyphen.
                           hyphenationLang="en_US"),
    "h2": ParagraphStyle("h2", fontName="Times-Bold", fontSize=12, leading=14,
                         spaceBefore=10, spaceAfter=6, keepWithNext=True),
    "h3": ParagraphStyle("h3", fontName="Times-Bold", fontSize=10.1, leading=12,
                         spaceBefore=7, spaceAfter=4, keepWithNext=True),
    "caption": ParagraphStyle("caption", fontName="Times-Italic", fontSize=8.2,
                              leading=10, spaceAfter=7),
    "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=7.3, leading=8.5),
    "ref": ParagraphStyle("ref", fontName="Times-Roman", fontSize=8.3, leading=10,
                          spaceAfter=5),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=6.5, leading=8.5,
                           spaceAfter=7),
}


def code(text):
    """Inline code: never hyphenated; a long token may break after / _ . ( or a comma.

    The break is a space too small to see, since the paragraph breaks only at
    spaces. Copying such a token out of the PDF picks the space up.
    """
    parts = re.split(r"(?<=[/_.(,])(?=[^\s)])", text) if len(text) > 12 else [text]
    return '<font size="0.1"> </font>'.join(
        '<nobr><font name="Courier" size="8">%s</font></nobr>' % part for part in parts if part)


def inline(node):
    text = (node.text or "") + "".join(ET.tostring(c, encoding="unicode") for c in node)
    text = re.sub(r'<code>(.*?)</code>', lambda m: code(m.group(1)), text)
    text = text.replace("<strong>", "<b>").replace("</strong>", "</b>")
    text = text.replace("<em>", "<i>").replace("</em>", "</i>")
    text = re.sub(r'<a href="([^"]+)">', r'<a href="\1" color="#194c70">', text)
    return text


def page(canvas, doc):
    canvas.setTitle("rGPU: Streaming Remote GPU Execution Across CUDA and PyTorch Boundaries")
    canvas.setAuthor("Research draft")
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(42, 25, "rGPU | Research draft | October 15, 2026")
    canvas.drawRightString(570, 25, str(doc.page))
    if doc.page == 1:
        canvas.setFillColor(colors.black)
        canvas.setFont("Times-Bold", 20)
        canvas.drawCentredString(306, 744, "rGPU: Streaming Remote GPU Execution")
        canvas.drawCentredString(306, 720, "Across CUDA and PyTorch Boundaries")
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(306, 699, "Systems paper draft | Preliminary evaluation | October 15, 2026")
    else:
        canvas.drawString(42, 765, "rGPU: Streaming Remote GPU Execution")
        canvas.setStrokeColor(colors.HexColor("#cccccc"))
        canvas.line(42, 758, 570, 758)


def frames(top):
    return [Frame(x, 42, width, top - 42, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0) for x in (42, 42 + width + 18)]


tree = ET.fromstring("<root>" + markdown.markdown(SOURCE, extensions=["tables", "fenced_code"]) + "</root>")
story = []
references = False
for node in tree:
    if node.tag == "h1" or (node.tag == "p" and (node.text or "").startswith("Systems paper draft")):
        continue
    if node.tag in ("h2", "h3"):
        if node.text == "References":
            references = True
        elif (node.text or "").startswith("Appendix"):
            references = False
        story.append(Paragraph(inline(node), styles[node.tag]))
    elif node.tag == "table":
        rows = [[Paragraph(inline(cell), styles["cell"]) for cell in tr]
                for tr in node.findall(".//tr")]
        n = len(rows[0])
        if n == 6:
            widths = [width * v for v in (.30, .12, .14, .16, .16, .12)]
        elif n == 5:
            widths = [width * v for v in (.35, .16, .29, .12, .08)]
        elif n == 4:
            widths = [width / 4] * 4
        else:
            widths = [width * v for v in (.33, .33, .34)]
        table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f4")),
            ("LINEABOVE", (0, 0), (-1, 0), .7, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), .4, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), .7, colors.black),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        table.keepWithNext = True
        gap = Spacer(1, 5)
        gap.keepWithNext = True
        story.extend([Spacer(1, 5), table, gap])
    elif node.tag == "pre":
        story.append(Preformatted("".join(node.itertext()).rstrip(), styles["code"], maxLineLength=60))
    elif node.tag == "p":
        img = node.find("img")
        if img is not None:
            im = Image(str(ROOT / img.attrib["src"]))
            im.drawHeight *= width / im.drawWidth
            im.drawWidth = width
            im.keepWithNext = True
            gap = Spacer(1, 4)
            gap.keepWithNext = True
            story.extend([Spacer(1, 5), im, gap])
        else:
            text = inline(node)
            kind = "caption" if re.match(r"^(Table|Figure) \d+\.", text) else "ref" if references else "body"
            story.append(Paragraph(text, styles[kind]))
    else:
        raise ValueError(f"Unhandled Markdown element: {node.tag}")

pdf_dir = ROOT
doc = BaseDocTemplate(str(pdf_dir / "rgpu-paper.pdf"), pagesize=(612, 792))
doc.addPageTemplates([PageTemplate(id="first", frames=frames(678), onPage=page,
                                   autoNextPageTemplate="later"),
                      PageTemplate(id="later", frames=frames(746), onPage=page)])
# Justified lines spread too thin. reportlab decides, line by line, how much
# space to add between words; watching the two functions that apply it gives
# the exact figure for every line drawn, which measuring the finished PDF can
# only estimate. They are private to reportlab, so a release that renames
# them turns this report off and leaves the rendering alone.
from reportlab.platypus import paragraph as _paragraph

STRETCH_LIMIT = 3.0    # widest acceptable gap, in normal spaces
justified = []


def _watch(name, unpack):
    drawer = getattr(_paragraph, name, None)
    if drawer is None:
        print(f"note: reportlab has no {name}; stretched lines are not reported")
        return

    def watching(tx, offset, *args):
        extra, spaces, unjustified, size, text, code_only = unpack(tx, *args)
        if spaces > 0 and extra > 1e-8 and not unjustified:
            normal = size / 4                  # a space in Times is a quarter em
            gap = (normal + extra / spaces) / normal
            justified.append((tx._canvas.getPageNumber(), gap, text, code_only))
        return drawer(tx, offset, *args)
    setattr(_paragraph, name, watching)


def _plain(tx, extra, words, last=0):
    text = " ".join(words)
    spaces = len(words) + _paragraph._nbspCount(text) - 1
    return extra, spaces, last, tx._fontsize or styles["body"].fontSize, text, False


def _rich(tx, line, last=0):
    words = [w for w in line.words if not hasattr(w, "cbDefn")]
    spaces = line.wordCount + sum(_paragraph._nbspCount(w.text) for w in words) - 1
    code_only = all(w.fontName == "Courier" for w in words if w.text.strip())
    return (line.extraSpace, spaces, last or line.lineBreak, styles["body"].fontSize,
            "".join(w.text for w in words), code_only)


_watch("_justifyDrawParaLine", _plain)
_watch("_justifyDrawParaLineX", _rich)

doc.build(story)
stretched = sorted(j for j in justified if j[1] > STRETCH_LIMIT)
for page, gap, text, _ in stretched:
    print("stretched: p%-2d gap %.1f spaces  %s" % (page, gap, text[:90]))
print(len(stretched), "of", len(justified),
      "justified lines are wider than %.0f normal spaces" % STRETCH_LIMIT)
# Every justified line, for stretched_lines.py --check to test itself against.
(ROOT / "figures" / "justified_lines.json").write_text(json.dumps(
    [{"page": pg, "gap": round(g, 3), "text": t, "code_only": c}
     for pg, g, t, c in sorted(justified)], indent=1))
