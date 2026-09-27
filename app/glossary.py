"""Glosar medical: promptarea Whisper + corectarea deterministă a termenilor (fără LLM, fără halucinații)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import yaml
from rapidfuzz import fuzz, process

from .config import get_config, resolve

CATEGORY_BY_MEETING = {
    "medical": ["intensive_care", "cardiology", "infectious", "surgery_urology", "pharmacology", "laboratory_imaging"],
    "executive": ["administrative", "pharmacology"],
    "administrative": ["administrative"],
}

_WORD = re.compile(r"[\w\-]+", re.UNICODE)


def _fold(s: str) -> str:
    """ș/ş, ț/ţ, diacritice -> formă comparabilă."""
    s = s.lower().replace("ş", "ș").replace("ţ", "ț")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


@dataclass
class Glossary:
    prompt_seed: dict
    categories: dict
    variants: dict
    _vocab: dict = field(default_factory=dict)  # lang -> {folded_word: canonical_word}

    def terms(self, lang: str, meeting_type: str | None = None) -> list[str]:
        cats = CATEGORY_BY_MEETING.get(meeting_type or "", list(self.categories))
        out: list[str] = []
        for c in cats:
            out += self.categories.get(c, {}).get(lang, [])
        return out

    def whisper_prompt(self, lang: str, meeting_type: str | None = None, max_chars: int = 380) -> str:
        """Prompt scurt (Whisper acceptă ~224 tokeni) — forma de listă reduce riscul ca promptul să fie „copiat”."""
        seed = self.prompt_seed.get(lang, "")
        terms = self.terms(lang, meeting_type)
        text = seed
        for t in terms:
            if len(text) + len(t) + 2 > max_chars:
                break
            text += ("" if not text else " " if text.endswith((".", " ")) else ", ") + t
        return text

    def hotwords(self, meeting_type: str | None = None) -> list[str]:
        return [t for lang in ("ro", "ru", "en") for t in self.terms(lang, meeting_type)]

    def vocab(self, lang: str) -> dict:
        if lang not in self._vocab:
            words: dict[str, str] = {}
            for cat in self.categories.values():
                for term in cat.get(lang, []):
                    for w in _WORD.findall(term):
                        if len(w) >= 7:
                            words[_fold(w)] = w
            self._vocab[lang] = words
        return self._vocab[lang]

    def correct(self, text: str, lang: str, threshold: int) -> tuple[str, list[tuple[str, str]]]:
        """Corectează termeni medicali deformați de ASR. Returnează textul + lista de corecturi (pentru audit)."""
        changes: list[tuple[str, str]] = []

        # 1) variante cunoscute (sintagme întâi, apoi cuvinte), potrivire exactă pe graniță de cuvânt
        for wrong in sorted(self.variants, key=len, reverse=True):
            right = self.variants[wrong]
            if wrong == right:
                continue
            pat = re.compile(rf"(?<![\w-]){re.escape(wrong)}(?![\w-])", re.IGNORECASE)
            if pat.search(text):
                def rep(m, right=right):
                    changes.append((m.group(0), right))
                    return right
                text = pat.sub(rep, text)

        # 2) fuzzy pe cuvinte lungi, doar față de vocabularul medical al limbii segmentului
        vocab = self.vocab(lang)
        if not vocab:
            return text, changes
        keys = list(vocab)

        def fix(m: re.Match) -> str:
            w = m.group(0)
            if len(w) < 7:
                return w
            fw = _fold(w)
            if fw in vocab:
                return w
            best = process.extractOne(fw, keys, scorer=fuzz.ratio, score_cutoff=threshold)
            if not best:
                return w
            cand_key = best[0]
            # prima literă și lungimea apropiată — evităm „corectarea” cuvintelor obișnuite
            if cand_key[0] != fw[0] or abs(len(cand_key) - len(fw)) > 2:
                return w
            # diferă doar terminația => e o flexiune (volemic/volemică, respirație/respiratorie), nu o eroare
            common = len(_common_prefix(fw, cand_key))
            if common >= min(len(fw), len(cand_key)) - 4:
                return w
            cand = _match_case(vocab[cand_key], w)
            changes.append((w, cand))
            return cand

        text = _WORD.sub(fix, text)
        return text, changes


def _common_prefix(a: str, b: str) -> str:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]


def _match_case(canon: str, original: str) -> str:
    if canon[:1].isupper():  # nume de medicamente / bacterii
        return canon
    if original[:1].isupper():
        return canon[:1].upper() + canon[1:]
    return canon


@lru_cache(maxsize=1)
def get_glossary() -> Glossary:
    cfg = get_config()
    with open(resolve(cfg.glossary.path), encoding="utf-8") as f:
        g = yaml.safe_load(f)
    return Glossary(
        prompt_seed=g.get("prompt_seed", {}),
        categories=g.get("categories", {}),
        variants={str(k).lower(): str(v) for k, v in (g.get("variants") or {}).items()},
    )
