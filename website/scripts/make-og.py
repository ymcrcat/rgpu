#!/usr/bin/env python3
"""Build public/og.png, the social card, from the site's own hero artwork.

Two things make rgpu-architecture.png unusable as a card directly: it is
1400x700 where crawlers expect 1200x630, and it has a real alpha channel.
X and Facebook composite a transparent PNG against their own theme, so
artwork drawn for a dark background can land on white and look broken.
This flattens it onto the site's dark tone at the right size.

    python3 scripts/make-og.py        # run from website/
"""
from PIL import Image

W, H = 1200, 630
BG = (11, 15, 20, 255)
MARGIN_X, MARGIN_Y = 70, 90

art = Image.open('public/rgpu-architecture.png').convert('RGBA')
card = Image.new('RGBA', (W, H), BG)
scale = min((W - 2 * MARGIN_X) / art.width, (H - 2 * MARGIN_Y) / art.height)
new = art.resize((round(art.width * scale), round(art.height * scale)), Image.LANCZOS)
card.alpha_composite(new, ((W - new.width) // 2, (H - new.height) // 2))
card.convert('RGB').save('public/og.png', 'PNG', optimize=True)
print('wrote public/og.png', Image.open('public/og.png').size)
