"""Vocabular medical RO / RU din corpusuri text deschise (rulat o singură dată, în faza de pregătire).

Surse:
  - SiMoNERo / MoNERo (RACAI, CC BY-SA 4.0): ~4.700 de fraze medicale românești cu termeni adnotați
    ANAT (anatomie), CHEM (medicamente, substanțe), DISO (boli, simptome), PROC (proceduri)
    https://github.com/UniversalDependencies/UD_Romanian-SiMoNERo
  - Mykes/rus_med_dialogues (MIT): dialoguri medic-pacient în rusă
  - blinoff/medical_qa_ru_data: întrebări și răspunsuri medicale în rusă

Produce în models/lexicon/:
  ro_medical_forms.txt  — toate formele de cuvânt din textele medicale RO (flexionate: „tromboprofilaxia”, „renală”)
  ru_medical_forms.txt  — la fel pentru RU
  medical_terms.tsv     — termeni medicali (inclusiv sintagme) cu categoria și frecvența: baza corectării

    python -m training.build_medical_vocab
"""
from __future__ import annotations

import collections
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "medical_text"
OUT = ROOT / "models" / "lexicon"
WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)


def norm(w: str) -> str:
    return w.lower().replace("ş", "ș").replace("ţ", "ț")


def simonero():
    forms: collections.Counter[str] = collections.Counter()
    terms: collections.Counter[tuple[str, str]] = collections.Counter()
    for f in sorted(SRC.glob("simonero-*.conllu")):
        span: list[str] = []
        cat = None
        for line in open(f, encoding="utf-8"):
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 10 or not cols[0].isdigit():
                if span:
                    terms[(" ".join(span), cat)] += 1
                    span, cat = [], None
                continue
            tok = norm(cols[1])
            if WORD.fullmatch(tok):
                forms[tok] += 1
            m = re.search(r"BioNERLabel=([BI])-(\w+)", cols[9])
            if m and (m.group(1) == "B" or not span):
                if span:
                    terms[(" ".join(span), cat)] += 1
                span, cat = [tok], m.group(2)
            elif m:
                span.append(tok)
            elif span:
                terms[(" ".join(span), cat)] += 1
                span, cat = [], None
    return forms, terms


def russian(max_rows: int = 150_000):
    import pyarrow.parquet as pq

    forms: collections.Counter[str] = collections.Counter()
    t = pq.read_table(SRC / "rus_med_dialogues.parquet")
    for col in ("user_question", "assistant_answer"):
        for x in t.column(col).to_pylist():
            forms.update(norm(w) for w in WORD.findall(x or ""))
    csv.field_size_limit(10 ** 9)
    with open(SRC / "medical_qa_ru.csv", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for i, row in enumerate(r):
            if i >= max_rows:
                break
            forms.update(norm(w) for w in WORD.findall((row.get("desc") or "") + " " + (row.get("ans") or "")))
    return forms


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ro_forms, ro_terms = simonero()
    (OUT / "ro_medical_forms.txt").write_text(
        "\n".join(f"{w} {c}" for w, c in ro_forms.most_common() if len(w) > 1), encoding="utf-8")

    ru_forms = russian()
    general_ru = {l.split()[0] for l in open(OUT / "ru_50k.txt", encoding="utf-8") if l.strip()}
    ru_keep = [(w, c) for w, c in ru_forms.most_common() if c >= 3 and len(w) > 1]
    (OUT / "ru_medical_forms.txt").write_text("\n".join(f"{w} {c}" for w, c in ru_keep), encoding="utf-8")

    # termeni RU „medicali” = frecvenți în textele medicale, dar absenți din vorbirea generală
    ru_terms = [(w, c) for w, c in ru_keep if c >= 15 and len(w) >= 6 and w not in general_ru]
    with open(OUT / "medical_terms.tsv", "w", encoding="utf-8") as fh:
        fh.write("lang\tterm\tcategory\tcount\n")
        for (t, cat), c in ro_terms.most_common():
            if len(t) >= 4:
                fh.write(f"ro\t{t}\t{cat}\t{c}\n")
        for w, c in ru_terms:
            fh.write(f"ru\t{w}\tMED\t{c}\n")
    cats = collections.Counter(cat for (_, cat) in ro_terms)
    print(f"RO: {len(ro_forms)} forme, {len(ro_terms)} termeni {dict(cats)} | "
          f"RU: {len(ru_keep)} forme, {len(ru_terms)} termeni medicali", file=sys.stderr)


if __name__ == "__main__":
    main()
