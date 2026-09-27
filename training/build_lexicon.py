"""Vocabularul românesc moldovenesc din transcrierile corpusului FraPiz/moldovan-dialectal-romanian-speech-corpus.

Adaugă la lexiconul RO formele regionale și cele din vorbirea spontană (dânsul, dumneavoastră, nu-i, ș.a.)
care lipsesc din listele de frecvență generale. Rulat o singură dată, în faza de dezvoltare.

    python -m training.build_lexicon
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "moldovan_corpus" / "metadata.jsonl"
DST = ROOT / "models" / "lexicon" / "ro_md_corpus.txt"


def main() -> None:
    cnt: collections.Counter[str] = collections.Counter()
    n = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"].lower().replace("ş", "ș").replace("ţ", "ț")
            cnt.update(re.findall(r"[^\W\d_]+", t))
            n += 1
    words = [(w, c) for w, c in cnt.most_common() if c >= 2 and len(w) > 1]
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text("\n".join(f"{w} {c}" for w, c in words), encoding="utf-8")
    print(f"{n} fraze, {len(cnt)} cuvinte unice, {len(words)} păstrate -> {DST}", file=sys.stderr)


if __name__ == "__main__":
    main()
