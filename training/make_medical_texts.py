"""Texte de antrenare pentru ASR: replici de ședință medicală în stil VORBIT moldovenesc, cu code-switching RO/RU/EN.

Generate LOCAL cu LLM-ul din Ollama (qwen3.5:9b), pornind de la glosarul medical al proiectului. Fiecare replică e o
listă de bucăți cu limba lor (ro/ru/en). Sinteza vocală (training/make_tts_xtts.py) le rostește pe fiecare în limba
ei, cu ACEEAȘI voce, apoi le lipește: rezultă code-switching real în interiorul frazei, cum vorbesc medicii.

Numerele sunt scrise în litere (așa se rostesc și așa sunt transcrierile corpusului moldovenesc); pipeline-ul le
transformă în cifre după ASR (app/ronum.py).

NU folosește transcrierea de referință Medpark (aceea rămâne exclusiv pentru evaluare).

    python -m training.make_medical_texts --n 2000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "medical_text" / "spoken_medical.jsonl"

SCENARIOS = [
    ("medical", "raportul de dimineață în terapie intensivă (ATI): pacienți pe paturi, parametri vitali, suport vasopresor, analize"),
    ("medical", "consiliu medical multidisciplinar: se decide tactica (operație, stent, tratament conservator), cine face ce și până când"),
    ("medical", "discuție de gardă despre un pacient care s-a agravat: tensiune, saturație, doze, ce s-a administrat"),
    ("medical", "raport despre antibioterapie: culturi, antibiogramă, doze ajustate la funcția renală"),
    ("medical", "discuție oncologică: histologie, chimioterapie, consultul oncologului, prognostic"),
    ("medical", "raport postoperator: drenaje, pansamente, mobilizare, tromboprofilaxie, analgezie"),
    ("executive", "ședință executivă a conducerii spitalului: buget, achiziții de echipament, termene, responsabili"),
    ("administrative", "ședință administrativă: grafic de gărzi, personal, stocuri de medicamente, raportări către minister"),
]
SPEC_HINT = {"cardiology": "cardiologie", "icu": "terapie intensivă", "infectious": "boli infecțioase",
             "urology": "urologie", "oncology": "oncologie", "neurology": "neurologie", "surgery": "chirurgie"}

SYSTEM = """Scrii replici REALISTE rostite la ședințele din spitalul Medpark din Chișinău (Republica Moldova).
Medicii vorbesc spontan, în română cu particularități moldovenești, și trec des în rusă sau engleză în mijlocul frazei.

Stil obligatoriu (vorbire, nu text scris):
- fraze vorbite, uneori neterminate, cu ezitări („deci”, „așa”, „păi”, „nu?”, „da, da”), 8-35 de cuvinte;
- regionalisme moldovenești reale acolo unde e natural: „dânsul/dânsa”, „amu”, „acuș”, „o fost” (= a fost),
  „îi” (= este), „aista/aiasta”, „nu-i”, „ghini”, dar nu în fiecare frază;
- NUMERELE SCRISE ÎN LITERE, exact cum se rostesc: „patul nouă”, „optzeci pe patruzeci”, „zero virgulă doi”,
  „o mie cinci sute de miligrame”, „pe data de optsprezece”. Niciodată cifre.
- termeni medicali corecți (denumiri de medicamente, proceduri, diagnostice), rostiți cum îi spun medicii;
- rusa se scrie DOAR cu litere chirilice, engleza cu litere latine; fiecare bucată are limba ei.
- fără nume reale de pacienți; numele de medici sunt nume de familie moldovenești obișnuite."""

PROMPT = """Context: {scenario}. Specialitate: {spec}.
Termeni de folosit (câte unul-doi pe replică, nu toți): {terms}

Scrie exact {k} replici, de la vorbitori diferiți, respectând pentru fiecare combinația de limbi cerută:
{slots}

„ro > ru > ro” = începe în română, trece în rusă în mijlocul frazei (o bucată scurtă, cum scapă medicii:
„ну короче”, „давление упало”, „в реанимации”, „по анализам всё норм”), apoi revine în română.
„ro > en” = fraza în română cu o bucată în engleză (termen sau propoziție: „follow-up”, „ejection fraction”,
„we need a CT scan”). Fiecare replică = listă de bucăți {{"lang": "ro"|"ru"|"en", "text": "..."}}, în ordinea rostirii."""

SLOTS = ["ro"] * 5 + ["ro > ru > ro", "ro > ru", "ru > ro", "ro > ru > ro", "ro > en", "ro > en > ro", "ru", "en"]

SCHEMA = {"type": "object", "properties": {"replici": {"type": "array", "minItems": 10, "items": {"type": "object", "properties": {
    "parts": {"type": "array", "items": {"type": "object", "properties": {
        "lang": {"type": "string", "enum": ["ro", "ru", "en"]}, "text": {"type": "string"}},
        "required": ["lang", "text"]}}}, "required": ["parts"]}}}, "required": ["replici"]}

CYR = re.compile(r"[а-яёА-ЯЁ]")
LAT = re.compile(r"[a-zA-ZăâîșțĂÂÎȘȚ]")


def valid(parts: list[dict]) -> list[dict] | None:
    out = []
    for p in parts:
        t = re.sub(r"\s+", " ", p.get("text", "")).strip()
        if not t:
            continue
        if re.search(r"\d", t):
            return None  # numerele trebuie rostite în litere
        if p["lang"] == "ru" and (LAT.search(t) or not CYR.search(t)):
            return None
        if p["lang"] in ("ro", "en") and CYR.search(t):
            return None
        out.append({"lang": p["lang"], "text": t})
    n = sum(len(p["text"].split()) for p in out)
    return out if out and 5 <= n <= 45 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--per-call", type=int, default=12)
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    gl = yaml.safe_load(open(ROOT / "config" / "medical_glossary.yaml", encoding="utf-8"))
    cats = {k: v for k, v in gl["categories"].items() if v.get("ro")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    have = sum(1 for _ in open(OUT, encoding="utf-8")) if OUT.exists() else 0
    client = httpx.Client(base_url="http://127.0.0.1:11434", timeout=600)
    t0, call = time.time(), 0
    with open(OUT, "a", encoding="utf-8") as f:
        while have < a.n:
            call += 1
            mtype, scen = rng.choice(SCENARIOS)
            ck = rng.choice(list(cats))
            terms = rng.sample(cats[ck]["ro"], min(8, len(cats[ck]["ro"])))
            for other in ("ru", "en"):
                if cats[ck].get(other):
                    terms += rng.sample(cats[ck][other], min(2, len(cats[ck][other])))
            plan = [rng.choice(SLOTS) for _ in range(a.per_call)]
            slots = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(plan))
            body = {"model": a.model, "stream": False, "format": SCHEMA, "think": False, "keep_alive": "2h",
                    "options": {"temperature": 0.95, "top_p": 0.95, "seed": rng.randint(0, 10**9),
                                "num_ctx": 4096, "num_predict": 2500},
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": PROMPT.format(scenario=scen, spec=SPEC_HINT.get(ck, ck),
                                                                          terms=", ".join(terms), k=a.per_call,
                                                                          slots=slots)}]}
            try:
                r = client.post("/api/chat", json=body).json()
                items = json.loads(r["message"]["content"]).get("replici", [])
            except Exception as e:  # noqa: BLE001
                print(f"apel {call}: eroare {e}", file=sys.stderr)
                continue
            ok = 0
            for it in items:
                parts = valid(it.get("parts", []))
                if parts:
                    f.write(json.dumps({"type": mtype, "spec": ck, "parts": parts}, ensure_ascii=False) + "\n")
                    ok += 1
            f.flush()
            have += ok
            print(f"apel {call}: +{ok}/{len(items)} -> {have}/{a.n} ({(time.time() - t0) / 60:.1f} min)",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
