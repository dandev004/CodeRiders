"""Numerale românești în cifre: „zero virgulă zero opt” -> „0,08”, „optzeci și șase” -> „86”, „patruzeci” -> „40”.

Modelul Whisper adaptat pe dialectul moldovenesc scrie numerele în litere (așa sunt transcrierile corpusului).
Într-un proces-verbal medical dozele, tensiunile și datele trebuie să fie în cifre — altfel „zero zero opt”
e ușor de citit greșit.
"""
from __future__ import annotations

import re

UNITS = {"zero": 0, "unu": 1, "una": 1, "un": 1, "o": 1, "doi": 2, "două": 2, "trei": 3, "patru": 4, "cinci": 5,
         "șase": 6, "șapte": 7, "opt": 8, "nouă": 9}
TEENS = {"zece": 10, "unsprezece": 11, "unșpe": 11, "doisprezece": 12, "douăsprezece": 12, "doișpe": 12,
         "treisprezece": 13, "treișpe": 13, "paisprezece": 14, "paișpe": 14, "cincisprezece": 15, "cincișpe": 15,
         "șaisprezece": 16, "șaișpe": 16, "șaptesprezece": 17, "șapteșpe": 17, "optsprezece": 18, "optișpe": 18,
         "nouăsprezece": 19, "nouășpe": 19}
TENS = {"douăzeci": 20, "douăzăci": 20, "treizeci": 30, "patruzeci": 40, "cincizeci": 50, "șaizeci": 60,
        "șaptezeci": 70, "optzeci": 80, "nouăzeci": 90}
HUNDRED = {"sută": 100, "sute": 100}
THOUSAND = {"mie": 1000, "mii": 1000}
ALL = set(UNITS) | set(TEENS) | set(TENS) | set(HUNDRED) | set(THOUSAND)
# cuvinte care arată ca numerale dar de regulă nu sunt (articol „o”, „un”) — convertite doar în context numeric
WEAK = {"o", "un", "una"}

_TOK = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)


def _value(words: list[str]) -> int | None:
    total, cur = 0, 0
    for w in words:
        if w in UNITS:
            cur += UNITS[w]
        elif w in TEENS:
            cur += TEENS[w]
        elif w in TENS:
            cur += TENS[w]
        elif w in HUNDRED:
            cur = max(cur, 1) * 100
        elif w in THOUSAND:
            total += max(cur, 1) * 1000
            cur = 0
        elif w == "și":
            continue
        else:
            return None
    return total + cur


def _is_digit_sequence(words: list[str]) -> bool:
    """„zero zero opt”, „trei opt” — cifre rostite una câte una."""
    ws = [w for w in words if w != "și"]
    return len(ws) >= 2 and all(w in UNITS and w not in WEAK for w in ws)


def _cls(w: str) -> str:
    return ("u" if w in UNITS else "t" if w in TEENS else "z" if w in TENS else "h" if w in HUNDRED
            else "k" if w in THOUSAND else "s")


# ce poate urma în interiorul UNUI număr: „trei sute”, „două mii”, „sută douăzeci și cinci”, „douăzeci și cinci”
_NEXT = {"u": {"h", "k"}, "h": {"u", "t", "z", "k"}, "k": {"u", "t", "z", "h"}, "z": {"s"}, "s": {"u"}, "t": {"h", "k"}}


def _groups(words: list[str]) -> list[list[str]]:
    """Împarte o secvență de numerale în numere separate: „trei opt patruzeci” = [3][8][40], nu 3+8+40=51."""
    out: list[list[str]] = []
    for w in words:
        c = _cls(w)
        if out and c in _NEXT.get(_cls(out[-1][-1]), set()):
            out[-1].append(w)
        else:
            out.append([w])
    return out


def _zero_led(digits: list[int]) -> str:
    """„zero zero opt” = 0,08 (doza rostită cifră cu cifră); „zero zero opt zero zero șapte” = 0,08 0,07."""
    groups, cur = [], []
    for k, d in enumerate(digits):
        # un nou număr începe la „zero” care urmează după o cifră nenulă
        if cur and d == 0 and cur[-1] != 0 and k + 1 < len(digits):
            groups.append(cur)
            cur = []
        cur.append(d)
    groups.append(cur)
    return " ".join("0," + "".join(map(str, g[1:])) if g[0] == 0 and len(g) > 1 else "".join(map(str, g))
                    for g in groups)


def _render(words: list[str]) -> str:
    if _is_digit_sequence(words):
        digits = [UNITS[w] for w in words if w != "și"]
        return _zero_led(digits) if digits[0] == 0 else "".join(map(str, digits))
    parts: list[str] = []
    digits = ""  # cifre rostite una câte una se lipesc: „trei opt” -> „38”
    for g in _groups(words):
        if len(g) == 1 and g[0] in UNITS and g[0] not in WEAK:
            digits += str(UNITS[g[0]])
            continue
        if digits:
            parts.append(digits)
            digits = ""
        v = _value(g)
        parts.append(str(v) if v is not None else " ".join(g))
    if digits:
        parts.append(digits)
    return " ".join(parts)


_NUMW = "|".join(sorted(ALL - WEAK, key=len, reverse=True))
# „pacientul patu' opt” (patul 8, pronunție moldovenească) e auzit de ASR ca „pacientul patru opt” și ar deveni „48”.
# În română 48 se rostește „patruzeci și opt”, niciodată „patru opt” — deci „patru” + numeral după „pacient”/„pe”
# înseamnă „patul”.
_BED = re.compile(rf"\b((?:pacient(?:ul|a)?|pe)\s+)patru(\s+(?:{_NUMW})\b)", re.IGNORECASE)


def to_digits(text: str) -> str:
    text = _BED.sub(lambda m: m.group(1) + "patul" + m.group(2), text)
    toks = _TOK.findall(text)
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        lw = t.lower()
        if lw in ALL:
            # adunăm o secvență de numerale (cu „și” și spații între ele)
            j, words = i, []
            while j < len(toks):
                x = toks[j].lower()
                # „și” leagă doar zeci + unități („douăzeci și cinci”), nu cifre („unu și opt” = 1 și 8)
                if x in ALL or (x == "și" and words and words[-1] in TENS and j + 2 < len(toks)
                                and toks[j + 2].lower() in UNITS):
                    words.append(x)
                    j += 1
                elif toks[j].isspace() and words:
                    j += 1
                else:
                    break
            while j > i and toks[j - 1].isspace():
                j -= 1
            if len(words) == 1 and (words[0] in WEAK or words[0] in HUNDRED or words[0] in THOUSAND):
                out.append(t)
                i += 1
                continue
            num = _render(words)
            # zecimale: „zero virgulă zero opt” -> „0,08”
            k = j
            while k < len(toks) and toks[k].isspace():
                k += 1
            if k < len(toks) and toks[k].lower() == "virgulă":
                m = k + 1
                frac: list[str] = []
                # partea zecimală: cifre rostite una câte una („zero opt”) SAU un singur numeral („douăzeci și doi”)
                while m < len(toks) and (toks[m].isspace() or toks[m].lower() in ALL or toks[m].lower() == "și"):
                    x = toks[m].lower()
                    if not toks[m].isspace():
                        if frac and (x not in UNITS or frac[-1] not in UNITS) and not (
                                x in UNITS and frac[-1] == "și") and not (x == "și" and frac[-1] in TENS):
                            break
                        frac.append(x)
                    m += 1
                while frac and frac[-1] == "și":
                    frac.pop()
                if frac:
                    fr = "".join(str(UNITS[w]) for w in frac) if all(w in UNITS for w in frac) else _render(frac)
                    num = f"{num},{fr}"
                    while m > k and toks[m - 1].isspace():
                        m -= 1
                    j = m
            out.append(num)
            i = j
        else:
            out.append(t)
            i += 1
    return "".join(out)
