"""Plauzibilitate lexicală: „este textul decodat un text real în limba respectivă?”

Problema măsurată pe înregistrarea Medpark: forțat pe rusă, Whisper transcrie româna moldovenească drept
„rusă fonetică” („шумблеры, воскулары” = „umplere vasculară”) cu o log-probabilitate CHIAR MAI BUNĂ decât
ipoteza românească. Scorul acustic singur nu poate decide limba. Criteriul suplimentar: ce proporție din
cuvintele ipotezei există în lexiconul limbii (50k cuvinte frecvente + glosarul medical + vocabularul
corpusului moldovenesc). Ipoteza în limba greșită produce cuvinte inexistente.

Cuvintele englezești sunt acceptate în ambele limbi (code-switching: „follow-up”, „CT scan”, „deadline”).
"""
from __future__ import annotations

import re
from functools import lru_cache

from .config import get_config, resolve
from .glossary import get_glossary

_W = re.compile(r"[^\W\d_]+", re.UNICODE)


def _norm(w: str) -> str:
    return w.lower().replace("ş", "ș").replace("ţ", "ț")


@lru_cache(maxsize=1)
def lexicons() -> dict[str, set[str]]:
    cfg = get_config().lexicon
    out: dict[str, set[str]] = {}
    for lang, files in cfg.files.items():
        words: set[str] = set()
        for f in files:
            p = resolve(f)
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    w = line.split(maxsplit=1)[0] if line.strip() else ""
                    if w:
                        words.add(_norm(w))
        out[lang] = words
    gl = get_glossary()
    for cat in gl.categories.values():
        for lang, terms in cat.items():
            for t in terms:
                out.setdefault(lang, set()).update(_norm(w) for w in _W.findall(t))
    return out


def words(text: str) -> list[str]:
    return [_norm(w) for w in _W.findall(text)]


def in_lexicon_rate(text: str, lang: str) -> tuple[float, int]:
    """(proporția cuvintelor recunoscute, numărul de cuvinte)."""
    ws = words(text)
    if not ws:
        return 0.0, 0
    lx = lexicons()
    own = lx.get(lang, set())
    en = lx.get("en", set())
    hit = sum(1 for w in ws if w in own or w in en)
    return hit / len(ws), len(ws)
