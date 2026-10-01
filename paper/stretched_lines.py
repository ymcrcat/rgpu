"""List justified lines in a PDF whose words are spread too far apart.

    python stretched_lines.py rgpu-paper.pdf
    python stretched_lines.py rgpu-paper.pdf --check figures/justified_lines.json
    python stretched_lines.py other.pdf --columns 1

A justified line that precedes an unbreakable long token (a commit hash, a
path) is left with few words and wide gaps. Exits 1 if any line's gaps exceed
LIMIT normal spaces.

render.py reports the same thing exactly, from the space reportlab added to
each line, and writes every justified line to figures/justified_lines.json.
This script measures the finished PDF instead, so it works on any PDF and
does not trust the renderer. --check compares the two line by line.

Two things it cannot do. A line made only of inline code is set smaller than
body text and is skipped with the table cells; the renderer's report covers
it. And where several gaps on a line follow inline code, whose space is wider
than the text's, the estimate runs up to one space high.

ponytail: assumes equal columns and Times-like body text. A different layout
needs --columns and BODY changed; per-block column detection if it ever has
to handle arbitrary PDFs.
"""
import collections
import json
import re
import statistics
import subprocess
import sys

LIMIT = 3.0      # widest acceptable gap, in normal spaces
COLUMNS = 2
BODY = (8.0, 8.8)   # word-box height of body text: cells and captions are smaller, headings larger
SPACE = 0.2778      # a normal space as a fraction of that height (a quarter em at 0.9 em tall)


def lines_of(pdf):
    """(page, gap in normal spaces, text) for every body line with two or more gaps."""
    tsv = subprocess.run(["pdftotext", "-tsv", pdf, "-"],
                         capture_output=True, text=True, check=True).stdout
    columns, width = collections.defaultdict(list), {}
    for row in tsv.splitlines()[1:]:
        f = row.split("\t")
        if len(f) < 12:
            continue
        page, left, top, w, h, text = int(f[1]), *map(float, f[6:10]), f[11]
        if f[0] == "1":
            width[page] = w
        elif f[0] == "5":
            columns[(page, int(left * COLUMNS / width[page]))].append(
                [top + h, left, left + w, h, text])
    out = []
    for (page, _), words in sorted(columns.items()):
        words.sort()
        rows, row = [], [words[0]]
        for w in words[1:]:
            # pdftotext's own line grouping breaks a stretched line into
            # one-word pieces, so group by baseline. Inline code sits about a
            # point off the text's.
            if w[0] - row[0][0] < 3:
                row.append(w)
            else:
                rows.append(row); row = [w]
        rows.append(row)
        for row in rows:
            row.sort(key=lambda w: w[1])
            merged = [row[0][:]]
            for w in row[1:]:
                # Punctuation after inline code is a word of its own with no
                # gap before it. It is part of the word it touches.
                if w[1] - merged[-1][2] < 0.5:
                    merged[-1][2], merged[-1][4] = w[2], merged[-1][4] + w[4]
                    merged[-1][3] = max(merged[-1][3], w[3])
                else:
                    merged.append(w[:])
            height = max(w[3] for w in merged)
            if len(merged) < 3 or not BODY[0] < height < BODY[1]:
                continue
            gaps = [b[1] - a[2] for a, b in zip(merged, merged[1:])]
            out.append((page, statistics.median(gaps) / (SPACE * height),
                        " ".join(w[4] for w in merged)))
    return out


def key(page, text):
    return page, re.sub(r"\W", "", text)[:40]


def check(found, truth_file):
    """Every line the renderer justified, against this script's estimate of it."""
    truth = {key(t["page"], t["text"]): t for t in json.load(open(truth_file))}
    mine = {key(page, text): gap for page, gap, text in found}
    errors, missed, wrong = [], [], []
    for k, t in truth.items():
        if t["text"].count(" ") < 2:
            continue                  # one gap: nothing to take a median of
        if t.get("code_only"):
            # Set smaller than body text, like a table cell, so by height
            # alone it cannot be told from one. The renderer's report has it.
            print("not measurable here, code only: p%d exact %.1f  %s"
                  % (t["page"], t["gap"], t["text"][:60]))
            continue
        if k not in mine:
            missed.append(t)
            continue
        errors.append(abs(mine[k] - t["gap"]))
        if (mine[k] > LIMIT) != (t["gap"] > LIMIT) and abs(t["gap"] - LIMIT) > 0.5:
            wrong.append((t, mine[k]))
    print(len(truth), "justified lines from the renderer;", len(errors), "matched")
    print("estimate against exact gap: median error %.2f spaces, worst %.2f"
          % (statistics.median(errors), max(errors)))
    for t in missed:
        print("NOT FOUND in the PDF: p%d gap %.1f  %s" % (t["page"], t["gap"], t["text"][:80]))
    for t, g in wrong:
        print("DISAGREE: p%d exact %.1f, estimated %.1f  %s"
              % (t["page"], t["gap"], g, t["text"][:80]))
    stretched = [t for t in truth.values() if t["gap"] > LIMIT]
    caught = [t for t in stretched if mine.get(key(t["page"], t["text"]), 0) > LIMIT - 0.5]
    print("lines the renderer stretched past %.0f spaces: %d, of which found here: %d"
          % (LIMIT, len(stretched), len(caught)))
    return not missed and not wrong and len(caught) == len(stretched)


def main():
    global COLUMNS
    if "--columns" in sys.argv:
        COLUMNS = int(sys.argv[sys.argv.index("--columns") + 1])
    found = lines_of(sys.argv[1])
    if "--check" in sys.argv:
        sys.exit(0 if check(found, sys.argv[sys.argv.index("--check") + 1]) else 1)
    bad = sorted((g, p, t) for p, g, t in found if g > LIMIT)
    for gap, page, text in reversed(bad):
        print("p%-2d gap %.1f spaces  %s" % (page, gap, text[:100]))
    print(len(bad), "of", len(found), "body lines are wider than %.0f normal spaces" % LIMIT)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
