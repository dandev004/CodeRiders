"""Corectarea medicală a transcrierii (după ASR, înainte de extragerea deciziilor).

De ce: pe vorbirea spontană cu accent moldovenesc, Whisper deformează termenii medicali („trombospirație”,
„cordașă”, „gădisânge”) și produce cuvinte inexistente. Textul fără sens ajunge apoi la LLM și în procesul-verbal.

Cum (fără să lăsăm modelul să „rescrie” ședința):
  1. Cuvinte suspecte = inexistente în lexiconul limbii (general + moldovenesc + medical, din SiMoNERo/MoNERo
     și corpusurile medicale rusești), în fraze cu încredere acustică mică.
  2. Candidați fonetici: termeni medicali care SUNĂ la fel după o cheie fonetică adaptată pronunției moldovenești
     (ă/â/î, ce/ci/che/chi, consoane surde/sonore, dublări).
  3. LLM-ul local primește fraza + fraza anterioară și următoare + candidații și întoarce DOAR o listă de corecturi
     „din -> în” (nu textul rescris) — rapid și fără posibilitatea de a adăuga conținut.
  4. Fiecare corectură e validată determinist: „din” trebuie să existe în frază, iar „în” trebuie să fie apropiat
     fonetic de „din” (sau un termen din vocabularul medical). Restul e respins. Originalul rămâne în raw_text.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from functools import lru_cache

from rapidfuzz import fuzz, process

from .config import get_config, resolve
from .glossary import get_glossary
from .lexicon import in_lexicon_rate, lexicons, words

log = logging.getLogger("secure_mom.medcorrect")

# ------------------------------------------------------------------------------------------------
# Cheie fonetică (RO, cu particularitățile de pronunție moldovenești)
# ------------------------------------------------------------------------------------------------

_PH = [
    (r"che", "ke"), (r"chi", "ki"), (r"ghe", "ge"), (r"ghi", "gi"), (r"ce", "ce"), (r"ci", "ci"),
    (r"[ăâî]", "a"), (r"ș", "s"), (r"ț", "t"), (r"x", "cs"), (r"y", "i"), (r"w", "v"), (r"q", "c"),
    (r"k", "c"), (r"ph", "f"), (r"th", "t"),
    (r"[bp]", "p"), (r"[dt]", "t"), (r"[gc]", "c"), (r"[zs]", "s"), (r"[vf]", "f"), (r"j", "s"),
    (r"e", "i"), (r"o", "u"),  # vocale închise în pronunția moldovenească (e->i, o->u în poziții neaccentuate)
    (r"(.)\1+", r"\1"),
]


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn" or c in "̆̂")


def phon(w: str) -> str:
    w = w.lower().replace("ş", "ș").replace("ţ", "ț")
    for a, b in _PH:
        w = re.sub(a, b, w)
    return w


@lru_cache(maxsize=1)
def medical_vocab() -> dict[str, dict[str, str]]:
    """lang -> {cheie_fonetică: formă_canonică} pentru termenii medicali (cuvinte >= 5 litere)."""
    out: dict[str, dict[str, str]] = {"ro": {}, "ru": {}, "en": {}}
    counts: dict[tuple[str, str], int] = {}
    p = resolve(get_config().medcorrect.get("terms_file", "models/lexicon/medical_terms.tsv"))
    if p.exists():
        for line in open(p, encoding="utf-8").read().splitlines()[1:]:
            lang, term, _, c = line.split("\t")
            for w in term.split():
                if len(w) >= 5:
                    counts[(lang, w)] = counts.get((lang, w), 0) + int(c)
    for cat in get_glossary().categories.values():
        for lang, terms in cat.items():
            for t in terms:
                for w in re.findall(r"[^\W\d_]+", t):
                    if len(w) >= 5:
                        counts[(lang, w.lower())] = counts.get((lang, w.lower()), 0) + 1000  # glosarul are prioritate
    for (lang, w), _c in sorted(counts.items(), key=lambda x: x[1]):
        out.setdefault(lang, {})[phon(w) if lang == "ro" else w] = w
    return out


def candidates(word: str, lang: str, k: int = 3) -> list[str]:
    voc = medical_vocab().get(lang, {})
    if not voc or len(word) < 5:
        return []
    key = phon(word) if lang == "ro" else word.lower()
    res = process.extract(key, list(voc), scorer=fuzz.ratio, limit=k, score_cutoff=78)
    return [voc[r[0]] for r in res if abs(len(r[0]) - len(key)) <= 3]


def suspicious_words(text: str, lang: str) -> list[str]:
    lx = lexicons()
    own, en = lx.get(lang, set()), lx.get("en", set())
    out = []
    for w in re.findall(r"[^\W\d_]+(?:-[^\W\d_]+)*", text):
        lw = w.lower().replace("ş", "ș").replace("ţ", "ț")
        if len(lw) >= 4 and lw not in own and lw not in en and lw.replace("-", "") not in own:
            out.append(w)
    return out


# ------------------------------------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------------------------------------

SYSTEM = """Ești corector al transcrierilor automate ale ședințelor medicale din spitalul Medpark (Chișinău).
Vorbitorii folosesc româna cu accent și regionalisme moldovenești, termeni medicali și uneori rusă/engleză.
Recunoașterea vocală a deformat unele cuvinte. Sarcina ta: găsește cuvintele GREȘIT RECUNOSCUTE și propune forma
corectă, pe baza contextului medical și a sunetului.

Reguli stricte:
- Corectezi doar cuvinte (sau grupuri scurte de 1-3 cuvinte) care sunt evident greșit recunoscute: cuvinte
  inexistente, termeni medicali deformați, forme fără sens în context.
- Corectura trebuie să SUNE aproape la fel ca originalul (e o eroare de auz, nu de sens).
- NU reformula, NU schimba stilul vorbit, NU corecta regionalismele reale (dânsul, acuș, amu, goluță), NU adăuga
  cuvinte sau informații, NU schimba numerele.
- Dacă nu ești sigur, nu corecta. Mai bine o frază neclară decât una inventată.
- Folosește candidații din vocabularul medical doar dacă se potrivesc cu sunetul și contextul."""

PROMPT = """Tipul ședinței: {meeting_type}. Termeni frecvenți în astfel de ședințe: {hints}

Fraze de corectat (cu fraza anterioară/următoare doar pentru context):
{items}

Răspunde în JSON: pentru fiecare frază id-ul și lista de corecturi {{"from": fragment exact din frază, "to": forma corectă}}.
Listă goală dacă fraza nu are erori evidente."""

SCHEMA = {"type": "object", "properties": {"fixes": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "edits": {"type": "array", "items": {
        "type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
        "required": ["from", "to"]}}}, "required": ["id", "edits"]}}}, "required": ["fixes"]}


def _valid_edit(src: str, dst: str, text: str, lang: str) -> bool:
    if not src or not dst or src == dst or src not in text:
        return False
    if len(src.split()) > 4 or len(dst.split()) > 5:
        return False
    if re.search(r"\d", src + dst) and re.sub(r"\D", "", src) != re.sub(r"\D", "", dst):
        return False  # numerele nu se ating
    a, b = (phon(src), phon(dst)) if lang == "ro" else (src.lower(), dst.lower())
    sim = fuzz.ratio(a, b)
    in_vocab = any(w.lower() in lexicons().get(lang, set()) for w in words(dst))
    return sim >= 70 or (sim >= 55 and in_vocab)


def correct_segments(segments: list[dict], meeting_type: str,
                     progress=None) -> tuple[list[dict], dict]:
    from .llm import Ollama

    cfg = get_config().medcorrect
    t0 = time.time()
    stats = {"checked": 0, "sent": 0, "edits_proposed": 0, "edits_applied": 0}
    if not cfg.get("enabled", True):
        return segments, stats
    # care fraze merită trimise: cuvinte inexistente sau încredere acustică mică
    todo = []
    for i, s in enumerate(segments):
        stats["checked"] += 1
        if s["lang"] not in ("ro", "ru"):
            continue
        sus = suspicious_words(s["text"], s["lang"])
        rate, n = in_lexicon_rate(s["text"], s["lang"])
        if sus or (n >= 3 and rate < 0.85) or s.get("confidence", 1) < float(cfg.get("confidence_below", 0.55)):
            todo.append((i, sus))
    stats["sent"] = len(todo)
    if not todo:
        return segments, stats

    gl = get_glossary()
    hints = ", ".join(gl.terms("ro", meeting_type)[:60])
    llm = Ollama()
    batch = int(cfg.get("batch_segments", 12))
    for b0 in range(0, len(todo), batch):
        part = todo[b0:b0 + batch]
        items = []
        for i, sus in part:
            s = segments[i]
            prev = segments[i - 1]["text"] if i > 0 else ""
            nxt = segments[i + 1]["text"] if i + 1 < len(segments) else ""
            cands = {w: candidates(w, s["lang"]) for w in sus}
            cands = {w: c for w, c in cands.items() if c}
            items.append({"id": i, "lang": s["lang"], "text": s["text"], "prev": prev[-160:], "next": nxt[:160],
                          "candidates": cands})
        try:
            out, _ = llm.chat_json(SYSTEM, PROMPT.format(meeting_type=meeting_type, hints=hints,
                                                         items=json.dumps(items, ensure_ascii=False, indent=0)), SCHEMA)
        except Exception as e:  # noqa: BLE001
            log.warning("Corectarea medicală a eșuat pentru un lot: %s", e)
            continue
        for fx in out.get("fixes", []):
            i = fx.get("id")
            if not isinstance(i, int) or i not in {p[0] for p in part}:
                continue
            s = segments[i]
            for ed in fx.get("edits", []):
                stats["edits_proposed"] += 1
                src, dst = ed.get("from", "").strip(), ed.get("to", "").strip()
                if _valid_edit(src, dst, s["text"], s["lang"]):
                    s.setdefault("raw_text", s["text"])
                    s["text"] = s["text"].replace(src, dst, 1)
                    s.setdefault("corrections", []).append({"from": src, "to": dst, "by": "llm"})
                    stats["edits_applied"] += 1
        if progress:
            progress(min(1.0, (b0 + len(part)) / len(todo)), f"corectare medicală {b0 + len(part)}/{len(todo)}")
    stats["seconds"] = round(time.time() - t0, 1)
    return segments, stats


# ------------------------------------------------------------------------------------------------
# Corectare deterministă (fără LLM) — ACTIVĂ în pipeline
# Măsurat pe transcrierea de referință Medpark: WER 52.1% -> 51.0%, termeni medicali recunoscuți 68% -> 77%.
# ------------------------------------------------------------------------------------------------

def _fold(w: str) -> str:
    w = w.lower().replace("ş", "ș").replace("ţ", "ț")
    return "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=1)
def _diacritic_index() -> dict[str, str]:
    """formă fără diacritice -> forma cu diacritice cea mai frecventă (din lexiconul RO + textele medicale)."""
    best: dict[str, tuple[int, str]] = {}
    cfg = get_config().lexicon
    for f in cfg.files.get("ro", []):
        p = resolve(f)
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            parts = line.split()
            if len(parts) < 1:
                continue
            w = parts[0].lower()
            c = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            k = _fold(w)
            if k != w and (k not in best or c > best[k][0]):
                best[k] = (c, w)
    return {k: w for k, (_, w) in best.items()}


@lru_cache(maxsize=1)
def _general_ro() -> set[str]:
    """Cuvintele uzuale (lista de frecvență generală) — nu sunt ținte bune pentru înlocuire."""
    out: set[str] = set()
    for f in get_config().lexicon.files.get("ro", [])[:1]:
        p = resolve(f)
        if p.exists():
            out |= {line.split()[0].lower() for line in open(p, encoding="utf-8") if line.strip()}
    return out


@lru_cache(maxsize=1)
def _medical_forms() -> dict[str, str]:
    """cheie fonetică -> formă medicală (termeni + forme flexionate din SiMoNERo/MoNERo + glosar).

    Formele din SiMoNERo care sunt și cuvinte uzuale („zonele”, „articol”) sunt excluse: un cuvânt inexistent
    înlocuit cu un cuvânt uzual care sună asemănător („sundele” -> „zonele”, în loc de „tensiunile”) strică sensul.
    Termenii din glosar și entitățile medicale adnotate rămân mereu."""
    out = dict(medical_vocab().get("ro", {}))
    common = _general_ro()
    p = resolve("models/lexicon/ro_medical_forms.txt")
    if p.exists():
        for line in open(p, encoding="utf-8"):
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) >= 6 and int(parts[1]) >= 2 and parts[0].lower() not in common:
                out.setdefault(phon(parts[0]), parts[0])
    return out


@lru_cache(maxsize=1)
def _critical_terms() -> set[str]:
    """Medicamente/substanțe (CHEM), proceduri dintr-un cuvânt și termenii de glosar dintr-un singur cuvânt.

    Nu spargem expresiile („transfer pe secție” nu face din „transfer” un termen critic) și nu includem categoriile
    generale (DISO/ANAT conțin și „risc”, „efort”) — altfel jumătate din transcriere ar fi „critică”."""
    out: set[str] = set()
    p = resolve(get_config().medcorrect.get("terms_file", "models/lexicon/medical_terms.tsv"))
    if p.exists():
        for line in open(p, encoding="utf-8").read().splitlines()[1:]:
            lang, term, cat, _ = line.split("\t")
            if " " not in term and (cat == "CHEM" or (cat == "PROC" and len(term) >= 7)):
                out.add(term.lower())
    for cat in get_glossary().categories.values():
        for terms in cat.values():
            out |= {t.lower() for t in terms if " " not in t.strip() and len(t) >= 4}
    return out


def is_critical(word: str, lang: str = "ro") -> bool:
    """Numere (doze, paturi, tensiuni, date) și medicamente/proceduri — ce un proces-verbal nu are voie să greșească."""
    w = word.strip(",.;:!?()\"'„”«»").lower()
    if not w:
        return False
    return bool(re.search(r"\d", w)) or w in _critical_terms()


def phonetic_fix(text: str, lang: str, threshold: int = 92) -> tuple[str, list[dict]]:
    """1) restaurează diacriticele lipsă; 2) cuvânt INEXISTENT -> termenul medical care sună aproape identic."""
    if lang != "ro":
        return text, []
    lx = lexicons()
    own, en = lx.get("ro", set()), lx.get("en", set())
    dia = _diacritic_index()
    forms = _medical_forms()
    keys = list(forms)
    changes: list[dict] = []

    def rep(m: re.Match) -> str:
        w = m.group(0)
        lw = w.lower()
        if len(lw) < 4 or lw in own or lw in en:
            return w
        new = None
        if _fold(lw) == lw and lw in dia:
            new = dia[lw]
        elif len(lw) >= 6:
            r = process.extractOne(phon(lw), keys, scorer=fuzz.ratio, score_cutoff=threshold)
            if r and abs(len(forms[r[0]]) - len(lw)) <= 2:
                cand = forms[r[0]]
                pre = 0
                while pre < min(len(cand), len(lw)) and _fold(cand)[pre] == _fold(lw)[pre]:
                    pre += 1
                # diferă doar terminația = altă flexiune a aceluiași cuvânt (anemica/anemia) -> nu atingem;
                # începutul diferit („mărtigul” -> „articol”) = alt cuvânt, nu o eroare de auz -> nu atingem
                if pre < min(len(cand), len(lw)) - 3 and phon(cand[:1]) == phon(lw[:1]):
                    new = cand
        if not new or new == lw:
            return w
        if w[:1].isupper():
            new = new[:1].upper() + new[1:]
        changes.append({"from": w, "to": new, "by": "fonetic"})
        return new

    return re.sub(r"[^\W\d_]+", rep, text), changes
