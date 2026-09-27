"""LLM local (Ollama) — de la transcriere brută la procesul-verbal structurat (MoM).

Principii împotriva halucinațiilor:
  - output JSON forțat prin schemă (Ollama structured outputs), temperatură mică;
  - fiecare decizie / sarcină trebuie să vină cu un CITAT-DOVADĂ copiat din transcriere + marcaj de timp;
    citatul e verificat fuzzy în Python — ce nu se regăsește în transcriere e eliminat sau marcat „de verificat”;
  - termenele relative („până vineri”, „mâine”) sunt rezolvate față de data ședinței, validate ca dată ISO;
  - transcrieri lungi: map (extragere pe bucăți) -> reduce (deduplicare + sinteză), totul local.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from typing import Callable

import httpx
from rapidfuzz import fuzz

from .config import get_config

log = logging.getLogger("secure_mom.llm")

LANG_NAMES = {"ro": "română (limba de stat a Republicii Moldova)", "ru": "rusă", "en": "engleză"}
WEEKDAYS_RO = ["luni", "marți", "miercuri", "joi", "vineri", "sâmbătă", "duminică"]

# ------------------------------------------------------------------------------------------------
# Scheme JSON
# ------------------------------------------------------------------------------------------------

_STR = {"type": "string"}
_NSTR = {"type": ["string", "null"]}

PARTICIPANT = {"type": "object", "properties": {"speaker": _STR, "name": _NSTR, "role": _NSTR},
               "required": ["speaker", "name", "role"]}
DECISION = {"type": "object", "properties": {
    "decision": _STR, "decided_by": _NSTR, "subject": _NSTR, "rationale": _NSTR,
    "timestamp": _STR, "evidence_quote": _STR},
    "required": ["decision", "decided_by", "subject", "rationale", "timestamp", "evidence_quote"]}
ACTION = {"type": "object", "properties": {
    "task": _STR, "subject": _NSTR, "owner": _NSTR, "owner_speaker": _NSTR, "deadline_text": _NSTR, "deadline_date": _NSTR,
    "priority": {"type": "string", "enum": ["urgent", "high", "normal"]}, "timestamp": _STR, "evidence_quote": _STR},
    "required": ["task", "subject", "owner", "owner_speaker", "deadline_text", "deadline_date", "priority", "timestamp",
                 "evidence_quote"]}
TOPIC = {"type": "object", "properties": {"title": _STR, "subject": _STR, "summary": _STR, "timestamp": _STR},
         "required": ["title", "subject", "summary", "timestamp"]}
# confuzii reale din discuție (nu din ASR): „nu e clar dacă consultul e la oncolog sau la neurochirurg”
NOTE = {"type": "object", "properties": {"note": _STR, "subject": _NSTR, "timestamp": _STR, "evidence_quote": _STR},
        "required": ["note", "subject", "timestamp", "evidence_quote"]}

MAP_SCHEMA = {"type": "object", "properties": {
    "participants": {"type": "array", "items": PARTICIPANT},
    "topics": {"type": "array", "items": TOPIC},
    "decisions": {"type": "array", "items": DECISION},
    "action_items": {"type": "array", "items": ACTION},
    "open_issues": {"type": "array", "items": _STR},
    "verification_notes": {"type": "array", "items": NOTE}},
    "required": ["participants", "topics", "decisions", "action_items", "open_issues", "verification_notes"]}

MOM_SCHEMA = {"type": "object", "properties": {
    "title": _STR, "summary": _STR,
    "participants": {"type": "array", "items": PARTICIPANT},
    "topics": {"type": "array", "items": TOPIC},
    "decisions": {"type": "array", "items": DECISION},
    "action_items": {"type": "array", "items": ACTION},
    "open_issues": {"type": "array", "items": _STR},
    "verification_notes": {"type": "array", "items": NOTE},
    "next_meeting": _NSTR},
    "required": ["title", "summary", "participants", "topics", "decisions", "action_items", "open_issues",
                 "verification_notes", "next_meeting"]}

SUMMARY_SCHEMA = {"type": "object", "properties": {
    "title": _STR, "summary": _STR, "meeting_type": _STR,
    "open_issues": {"type": "array", "items": _STR},
    "next_meeting": _NSTR},
    "required": ["title", "summary", "meeting_type", "open_issues", "next_meeting"]}

# ------------------------------------------------------------------------------------------------
# Prompturi
# ------------------------------------------------------------------------------------------------

SYSTEM = """Ești secretarul unei ședințe din Spitalul Internațional Medpark (Chișinău, Republica Moldova).
Primești transcrierea automată a ședinței. Vorbitorii trec brusc între română (cu regionalisme moldovenești),
rusă și engleză, uneori în aceeași frază. Transcrierea poate conține mici erori de recunoaștere vocală.

Reguli stricte:
- Scrii TOT conținutul în limba {out_lang}. Înțelegi fragmentele în rusă/engleză și le redai în {out_lang}.
- Termenii medicali, denumirile de medicamente, bacterii, proceduri și abrevierile (ATI, CT, INR...) se scriu corect.
- NU inventa nimic. Dacă ceva nu reiese clar din transcriere, folosește null. Nu adăuga unități de măsură care nu
  au fost rostite, nu interpreta și nu scrie „probabil X” — o valoare neclară o treci în verification_notes.
- Un termen deformat de recunoașterea vocală îl corectezi DOAR dacă forma corectă e evidentă (ex. „ceftriaxonă”
  auzit „cefriaxonă”, „Pseudomonas” auzit „pseudomonaz”). Altfel îl lași cum e — fără presupuneri de tipul „probabil X”.
- DECIZIE = ceva ce s-a hotărât/aprobat/stabilit (o conduită, o schimbare de tratament, o aprobare, o regulă).
  La consiliul medical, ORICE modificare de tratament anunțată sau deja făcută e o decizie separată, cu valorile
  rostite: se administrează/reîncarcă volum, se scade/crește/oprește un vasopresor sau inotrop (NA, dobutamină),
  se comandă/transfuzează sânge, se schimbă/ajustează antibioticul, se montează un cateter/linie arterială,
  se indică o investigație sau un consult. Nu este decizie o simplă constatare sau o discuție fără concluzie.
- SARCINĂ (action item) = cineva trebuie să facă concret ceva după/în urma ședinței. Owner = persoana care
  preia sarcina (nume dacă e rostit, altfel funcția, ex. „medicul de gardă ATI”); owner_speaker = eticheta
  vorbitorului (S1, S2...) dacă din dialog reiese cine a preluat-o, altfel null.
- Termen: deadline_text = DOAR termenul rostit efectiv în ședință („mâine dimineață”, „до пятницы”), tradus;
  deadline_date = data YYYY-MM-DD calculată față de data ședinței. Dacă nu s-a rostit niciun termen: AMBELE null.
  NU deduce termene „standard” sau „implicite” și nu scrie explicații de tipul „nu s-a specificat”.
- Fii concis: subiecte în 1-2 propoziții, fără repetări; rationale doar dacă motivul a fost spus, altfel null.
- evidence_quote = fragmentul EXACT (copiat cuvânt cu cuvânt, în limba originală, 5-25 de cuvinte) din transcriere
  care susține decizia/sarcina. timestamp = marcajul de timp [hh:mm:ss] al acelui fragment.
- Pacienții se identifică doar cum apar în transcriere (ex. „pacientul de pe patul 4”) — nu adăuga date personale.
  Liniile „— se discută: patul N —” arată pacientul discutat în replicile care urmează.
- subject = CUI îi aparține informația: pentru un pacient „Patul N” dacă patul e rostit, altfel o descriere scurtă
  („pacientul cu hidronefroză”); pentru teme non-clinice tema („Grafic de gărzi”, „Achiziții”). Același pacient are
  EXACT același subject în toate subiectele, deciziile și sarcinile.
- Stil clinic, nu corporatist: abrevieri medicale standard (TA, SpO2, AV, FR, NA, FEVS, Hb, ECG, CT, ATI, IMA, i.v.),
  valorile concrete rostite (TA 80/40, Hb 86, creatinină 190, doze cu unitate). Fără cuvinte ca „livrabil”, „action item”.
- verification_notes = DOAR confuziile reale din discuție care pot schimba conduita (ex. nu e clar dacă se cheamă
  oncologul sau neurochirurgul; două doze diferite rostite pentru același medicament). Cu citat exact. Altfel listă goală.
- Participanți: pentru fiecare etichetă de vorbitor, numele doar dacă e rostit în ședință (cineva i se adresează
  sau se prezintă), iar rolul dedus din context (ex. „șef ATI”, „medic cardiolog”), altfel null."""

MAP_PROMPT = """Tipul ședinței: {meeting_type}. Data ședinței: {date} ({weekday}).
Aceasta este partea {part} din {parts} a transcrierii.{focus}

TRANSCRIERE:
{transcript}

Extrage din această parte: participanții (etichete de vorbitor); subiectele — câte unul pentru FIECARE pacient
discutat și pentru fiecare temă non-clinică; deciziile; sarcinile (cu responsabil și termen); problemele rămase
deschise; confuziile de verificat. Fiecare element are subject (pacientul sau tema).

Pentru un pacient, summary acoperă, cât s-a rostit, în 4-7 propoziții clinice:
1) diagnosticul principal, istoricul și intervențiile deja făcute (ex. angioplastie, tromboaspirație, operații);
2) parametrii vitali și analizele CU VALORILE rostite (TA, AV, SpO2, FEVS, Hb, lactat, creatinină, uree…);
3) evoluția și cauza ei: ce se ameliorează sau se agravează și de ce (ex. „după ajustarea X s-au redus dozele de Y”);
4) tratamentul curent și modificările lui (volum, vasopresoare/inotrope, antibiotice, transfuzii, stimulare).
Fiecare modificare de tratament (inclusiv o transfuzie comandată sau o doză redusă) este și o decizie separată.
Nu omite diagnosticul principal, modificările de tratament și dinamica parametrilor. Răspunde în JSON."""

FOCUS_BED = """
În această parte se discută pacientul de pe patul {bed}: diagnosticul, valorile, deciziile și sarcinile îi aparțin
lui (subject „Patul {bed}”), cu excepția celor pentru care se spune explicit alt pat. Pentru el scrie UN SINGUR
element în topics (situația clinică unitară); sarcinile, responsabilii și confuziile NU se scriu în topics."""

SUMMARY_PROMPT = """Tipul ședinței: {meeting_type}. Data ședinței: {date} ({weekday}). Durata: {duration}.

Mai jos sunt subiectele, deciziile și sarcinile deja extrase (și verificate) din ședință, în ordine cronologică.
Scrie în limba {out_lang}:
- title: titlu scurt și concret al ședinței (max 12 cuvinte);
- summary: rezumat executiv de 3-5 propoziții — ce s-a discutat și ce s-a hotărât;
- meeting_type: tipul ședinței după conținut, una dintre valorile: {type_choices};
- open_issues: problemele rămase nerezolvate (unite, fără duplicate); next_meeting: dacă s-a stabilit, altfel null.
Nu adăuga nimic ce nu apare în extrase.

{extracts}"""

TRANSLATE_PROMPT = """Tradu în limba {out_lang} TOATE valorile text din JSON-ul de mai jos (procesul-verbal al unei
ședințe medicale). Păstrează exact structura, cheile, marcajele de timp, datele YYYY-MM-DD, etichetele de vorbitor
(S1, S2...), valorile „priority”, numele proprii și denumirile de medicamente. Terminologia medicală trebuie să fie
cea corectă, folosită de medici în limba {out_lang}. Câmpul evidence_quote NU se traduce (rămâne citatul original).
Traduci fidel: NU adăuga informații, unități de măsură, doze sau explicații care nu sunt în textul sursă.

{payload}"""

# ------------------------------------------------------------------------------------------------


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def build_turns(segments: list[dict], max_gap: float = 2.0) -> list[dict]:
    """Frazele consecutive ale aceluiași vorbitor devin o singură replică (mai puțini tokeni, context mai clar)."""
    turns: list[dict] = []
    for s in segments:
        spk = s.get("speaker") or "S?"
        if turns and turns[-1]["speaker"] == spk and s["start"] - turns[-1]["end"] <= max_gap \
                and turns[-1]["end"] - turns[-1]["start"] < 90:
            t = turns[-1]
            t["text"] += " " + s["text"]
            t["word_conf"] = t.get("word_conf", []) + (s.get("word_conf") or [])
            t["end"] = s["end"]
            if s["lang"] not in t["langs"]:
                t["langs"].append(s["lang"])
        else:
            turns.append({"speaker": spk, "start": s["start"], "end": s["end"], "text": s["text"], "langs": [s["lang"]],
                          "word_conf": list(s.get("word_conf") or [])})
    return turns


def render_transcript(turns: list[dict]) -> str:
    """Replicile cu marcaj de timp; când se trece la alt pat, o linie „— se discută: patul N —” (calculată din
    transcriere, nu ghicită), ca LLM-ul să nu atribuie valorile și deciziile altui pacient."""
    out, cur = [], None
    for t in turns:
        if t.get("bed_ctx") and t["bed_ctx"] != cur:
            cur = t["bed_ctx"]
            out.append(f"— se discută: patul {cur} —")
        out.append(f"[{fmt_ts(t['start'])}] {t['speaker']} ({'/'.join(t['langs'])}): {t['text']}")
    return "\n".join(out)


def chunk_turns(turns: list[dict], max_chars: int) -> list[list[dict]]:
    chunks, cur, n = [], [], 0
    for t in turns:
        L = len(t["text"]) + 30
        if cur and n + L > max_chars:
            chunks.append(cur)
            # suprapunere de o replică: o decizie de la granița bucăților nu se pierde
            cur, n = cur[-1:], len(cur[-1]["text"]) + 30
        cur.append(t)
        n += L
    if cur:
        chunks.append(cur)
    return chunks


def chunk_by_patient(turns: list[dict], max_chars: int) -> list[tuple[str | None, list[dict]]]:
    """Bucățile urmează pacienții (patul discutat, din „patul N” rostit): un apel LLM vede un singur pacient, deci nu
    mută valori sau decizii între pacienți și nu scapă detaliile unui caz lung. Discuțiile scurte vecine se unesc
    (sub max_chars/2) ca o ședință cu mulți pacienți să nu ceară prea multe apeluri; cele lungi se taie la max_chars.
    Fără paturi rostite (ședință executivă/administrativă) = bucăți după mărime, ca înainte."""
    groups: list[tuple[str | None, list[dict]]] = []
    for t in turns:
        bed = t.get("bed_ctx")
        if groups and groups[-1][0] == bed:
            groups[-1][1].append(t)
        else:
            groups.append((bed, [t]))
    size = lambda ts: sum(len(x["text"]) + 30 for x in ts)  # noqa: E731
    merged: list[tuple[str | None, list[dict]]] = []
    for bed, g in groups:
        if merged and size(merged[-1][1]) + size(g) <= max_chars // 2:
            merged[-1] = (None, merged[-1][1] + g)  # mai mulți pacienți: marcajele din transcriere îi separă
        else:
            merged.append((bed, g))
    return [(bed, part) for bed, g in merged for part in chunk_turns(g, max_chars)]


class Ollama:
    def __init__(self):
        c = get_config().llm
        self.cfg = c
        self.client = httpx.Client(base_url=c.base_url, timeout=float(c.timeout_s))

    def available(self) -> tuple[bool, str]:
        try:
            tags = self.client.get("/api/tags", timeout=3).json()
            names = [m["name"] for m in tags.get("models", [])]
            ok = self.cfg.model in names or f"{self.cfg.model}:latest" in names
            return ok, "ok" if ok else f"modelul {self.cfg.model} nu e descărcat în Ollama ({', '.join(names)})"
        except Exception as e:  # noqa: BLE001
            return False, f"Ollama indisponibil: {e}"

    def chat_json(self, system: str, user: str, schema: dict, retries: int = 2) -> tuple[dict, dict]:
        body = {
            "model": self.cfg.model, "stream": False, "format": schema, "think": bool(self.cfg.think),
            "keep_alive": "30m",  # expirarea în timpul unei cereri blochează Ollama ~15 min (măsurat)
            "options": {"temperature": float(self.cfg.temperature), "seed": int(self.cfg.get("seed", 42)),
                        "num_ctx": int(self.cfg.num_ctx),
                        "num_predict": int(self.cfg.get("max_output_tokens", 3000)),
                        "top_p": 0.9, "repeat_penalty": 1.05},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        last = None
        for attempt in range(retries + 1):
            t0 = time.time()
            r = self.client.post("/api/chat", json=body)
            log.info("LLM call %.1fs (ollama %.1fs, load %.1fs)", time.time() - t0,
                     r.json().get("total_duration", 0) / 1e9 if r.is_success else -1,
                     r.json().get("load_duration", 0) / 1e9 if r.is_success else -1)
            r.raise_for_status()
            d = r.json()
            content = d["message"]["content"]
            try:
                data = json.loads(content)
                return data, {"prompt_tokens": d.get("prompt_eval_count", 0), "output_tokens": d.get("eval_count", 0),
                              "seconds": round(d.get("total_duration", 0) / 1e9, 1)}
            except json.JSONDecodeError as e:
                last = e
                log.warning("JSON invalid de la LLM (încercarea %d): %s", attempt + 1, content[:200])
                body["options"]["temperature"] = 0.3
        raise RuntimeError(f"LLM-ul nu a returnat JSON valid: {last}")

    def unload(self):
        try:
            self.client.post("/api/generate", json={"model": self.cfg.model, "keep_alive": 0}, timeout=10)
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------------------------------------
# Validare & post-procesare (deterministă)
# ------------------------------------------------------------------------------------------------

_TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})|(\d{1,2}):(\d{2})")


def _ts_seconds(ts: str | None) -> float | None:
    if not ts:
        return None
    m = _TS.search(ts)
    if not m:
        return None
    if m.group(1):
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    return int(m.group(4)) * 60 + int(m.group(5))


def verify_evidence(item: dict, turns: list[dict]) -> tuple[float, dict | None]:
    """Cât de bine se regăsește citatul-dovadă în transcriere (0..100) + replica sursă."""
    q = (item.get("evidence_quote") or "").strip().strip('"„”«»')
    if len(q) < 8:
        return 0.0, None
    ts = _ts_seconds(item.get("timestamp"))
    best, src = 0.0, None
    for t in turns:
        s = fuzz.partial_ratio(q.lower(), t["text"].lower())
        if ts is not None and abs(t["start"] - ts) < 120:
            s += 3  # mic bonus pentru replica din jurul marcajului de timp
        if s > best:
            best, src = s, t
    return min(100.0, best), src


def uncertain_critical(item: dict, src: dict | None) -> list[dict]:
    """Cuvintele CRITICE (numere, doze, termeni medicali) pe care ASR-ul nu a fost sigur, din fragmentul citat.

    Un proces-verbal medical nu poate „ghici” doza sau patul: dacă decizia se sprijină pe un astfel de cuvânt,
    documentul îl marchează explicit, cu momentul din înregistrare, ca să fie verificat la audio."""
    from .medcorrect import is_critical

    wc = (src or {}).get("word_conf") or []
    q = (item.get("evidence_quote") or "").strip().strip('"„”«»')
    if not wc or len(q) < 8:
        return []
    text = " ".join(x[0] for x in wc)
    al = fuzz.partial_ratio_alignment(q.lower(), text.lower())
    if al is None:
        return []
    lo = float(get_config().asr.get("word_confidence_low", 0.5))
    out, pos = [], 0
    for x in wc:
        w, p = x[0], x[1]
        start, pos = pos, pos + len(w) + 1
        if start + len(w) < al.dest_start or start > al.dest_end:
            continue
        if p is not None and p < lo and is_critical(w, (src.get("langs") or ["ro"])[0]):
            out.append({"word": w.strip(",.;:!?"), "p": round(p, 2)})
    return out


def _valid_date(s: str | None, meeting_date: dt.date) -> str | None:
    if not s:
        return None
    try:
        d = dt.date.fromisoformat(s.strip()[:10])
    except ValueError:
        return None
    # termene absurde (înainte de ședință sau peste 2 ani) — probabil halucinate
    if d < meeting_date or d > meeting_date + dt.timedelta(days=730):
        return None
    return d.isoformat()


_SUBJ_PREFIX = re.compile(r"^(?:pacient(?:ul|a)?|patient|пациент\w*|bolnav\w*)\s+(?:de pe |cu |din |with |с )?", re.I)


from .clinical import _BED as _BED_NUM  # noqa: E402


def assign_subjects(mom: dict, turns: list[dict] | None = None) -> None:
    """Grupare deterministă pe pacient/temă: cheia vine din numărul patului (regex), nu din formularea LLM-ului,
    ca „Patul 8”, „pacientul de pe patul 8” și „bed 8” să ajungă în același bloc al procesului-verbal.
    O decizie/sarcină fără pat în propriul text primește patul discutat în transcriere la momentul citatului
    (la consiliu pacienții se iau pe rând) — LLM-ul confundă uneori pacienții între ei."""
    from .clinical import bed_at, bed_of, bed_timeline, is_patient

    timeline = bed_timeline(turns or [])
    text_keys: dict[str, str] = {}

    def key_for(subject: str | None, *texts: str | None) -> tuple[str, str] | None:
        bed = bed_of(subject, *texts)
        if bed:
            return f"bed:{bed}", "patient"
        if not subject:
            return None
        norm = _SUBJ_PREFIX.sub("", subject.strip().lower()).strip(" .")
        hit = next((k for n, k in text_keys.items() if fuzz.token_set_ratio(norm, n) >= 85), None)
        k = hit or text_keys.setdefault(norm, f"txt:{norm}")
        return k, ("patient" if is_patient(subject) else "theme")

    topics = sorted(mom.get("topics", []), key=lambda x: _ts_seconds(x.get("timestamp")) or 0)
    for tp in topics:
        tp["subject_key"], tp["subject_kind"] = key_for(tp.get("subject"), tp.get("title"), tp.get("summary")) \
            or ("general", "theme")
    for key, field in (("decisions", "decision"), ("action_items", "task"), ("verification_notes", "note")):
        for it in mom.get(key, []):
            got = key_for(it.get("subject"), it.get(field))
            ctx = bed_at(timeline, _ts_seconds(it.get("timestamp")) or 0)
            quoted = bed_of(it.get("evidence_quote"))  # patul rostit chiar în citat are prioritate
            ctx = quoted or ctx
            if ctx and (got is None or got[1] == "patient"):
                got = f"bed:{ctx}", "patient"
                if bed_of(it.get(field)) not in (None, ctx):  # LLM-ul a scris alt pat decât cel discutat
                    it[field] = _BED_NUM.sub(lambda m: m.group(0)[:m.start(1) - m.start(0)] + ctx, it[field])
            if got is None:  # fără subiect: situația discutată cel mai recent înainte (max 3 min)
                ts = _ts_seconds(it.get("timestamp")) or 0
                prev = [t for t in topics if (_ts_seconds(t.get("timestamp")) or 0) <= ts + 5
                        and ts - (_ts_seconds(t.get("timestamp")) or 0) <= 180]
                got = (prev[-1]["subject_key"], prev[-1]["subject_kind"]) if prev else ("general", "theme")
            it["subject_key"], it["subject_kind"] = got


def postprocess(mom: dict, turns: list[dict], speakers: list[str], meeting_date: dt.date) -> dict:
    min_score = 72.0
    for key in ("decisions", "action_items", "verification_notes"):
        kept = []
        for it in mom.get(key, []):
            score, src = verify_evidence(it, turns)
            it["evidence_score"] = round(score)
            if src is not None:
                it["timestamp"] = fmt_ts(src["start"])
                it["source_speaker"] = src["speaker"]
                it["source_langs"] = src["langs"]
            if score < 55:
                log.info("Eliminat (%s) fără dovadă în transcriere: %s", key, it)
                continue
            it["verified"] = score >= min_score
            it["uncertain_terms"] = uncertain_critical(it, src)
            kept.append(it)
        mom[key] = sorted(kept, key=lambda x: _ts_seconds(x.get("timestamp")) or 0)

    for it in mom.get("action_items", []):
        if it.get("deadline_text") and _INVENTED_DEADLINE.search(it["deadline_text"]):
            it["deadline_text"], it["deadline_date"] = None, None
        it["deadline_date"] = _valid_date(it.get("deadline_date"), meeting_date)
        if it.get("owner_speaker") and it["owner_speaker"] not in speakers:
            it["owner_speaker"] = None

    # participanți: exact vorbitorii detectați de diarizare (nici mai mulți, nici mai puțini)
    by_spk = {p.get("speaker"): p for p in mom.get("participants", []) if p.get("speaker") in speakers}
    mom["participants"] = [by_spk.get(s) or {"speaker": s, "name": None, "role": None} for s in speakers]
    names = {p["speaker"]: p["name"] for p in mom["participants"] if p.get("name")}
    for it in mom.get("action_items", []):
        if not it.get("owner") and it.get("owner_speaker") in names:
            it["owner"] = names[it["owner_speaker"]]
    for i, it in enumerate(mom.get("decisions", []), 1):
        it["id"] = f"D{i}"
    for i, it in enumerate(mom.get("action_items", []), 1):
        it["id"] = f"A{i}"
    # unitățile de măsură adăugate de LLM, nerostite în ședință, se scot („TA 80/40 mmHg” -> „TA 80/40”)
    from .clinical import strip_unspoken_units
    spoken = " ".join(t.get("text") or "" for t in turns)
    for key, field in (("topics", "summary"), ("decisions", "decision"), ("action_items", "task")):
        for it in mom.get(key, []):
            it[field] = strip_unspoken_units(it.get(field), spoken)
    mom["summary"] = strip_unspoken_units(mom.get("summary"), spoken)
    assign_subjects(mom, turns)
    return mom


def _dedupe(items: list[dict], key: str) -> list[dict]:
    """Aceeași decizie menționată de mai multe ori (în bucăți diferite) = o singură intrare (cea mai bine susținută)."""
    out: list[dict] = []
    for it in items:
        txt = (it.get(key) or "").lower()
        dup = next((o for o in out if fuzz.token_set_ratio(txt, (o.get(key) or "").lower()) >= 85), None)
        if dup is None:
            out.append(it)
        elif len(it.get("evidence_quote") or "") > len(dup.get("evidence_quote") or ""):
            out[out.index(dup)] = it
    return out


def merge_parts(parts: list[dict]) -> dict:
    mom = {"participants": [], "topics": [], "decisions": [], "action_items": [], "open_issues": [],
           "verification_notes": [], "next_meeting": None}
    seen: dict[str, dict] = {}
    for p in parts:
        for x in p.get("participants", []):
            cur = seen.setdefault(x.get("speaker"), {"speaker": x.get("speaker"), "name": None, "role": None})
            cur["name"] = cur["name"] or x.get("name")
            cur["role"] = cur["role"] or x.get("role")
        mom["topics"] += p.get("topics", [])
        mom["decisions"] += p.get("decisions", [])
        mom["action_items"] += p.get("action_items", [])
        mom["open_issues"] += p.get("open_issues", [])
        mom["verification_notes"] += p.get("verification_notes", [])
    mom["participants"] = list(seen.values())
    mom["decisions"] = _dedupe(mom["decisions"], "decision")
    mom["action_items"] = _dedupe(mom["action_items"], "task")
    return mom


TYPE_HINTS = {
    "medical": " (pacienți, diagnostic, tratament, consulturi)",
    "executive": " (conducere: strategie, buget, indicatori, investiții)",
    "administrative": " (organizare: personal, gărzi, achiziții, logistică, IT)",
}

_INVENTED_DEADLINE = re.compile(r"implicit|standard|nu s-a (specificat|precizat|stabilit)|nespecificat|probabil|"
                                r"continuu|în continuare|la nevoie|dacă e necesar|not specified|не указан", re.IGNORECASE)


# ------------------------------------------------------------------------------------------------
# API
# ------------------------------------------------------------------------------------------------

def generate_mom(segments: list[dict], meeting_type: str, meeting_date: dt.date, speakers: list[str],
                 duration_s: float, progress: Callable[[float, str], None] | None = None) -> tuple[dict, dict]:
    cfg = get_config()
    out_lang = cfg.output.default_language
    llm = Ollama()
    ok, msg = llm.available()
    if not ok:
        raise RuntimeError(msg)

    turns = build_turns(segments)
    from .clinical import bed_at, bed_timeline
    timeline = bed_timeline(turns)
    for t in turns:
        t["bed_ctx"] = bed_at(timeline, t["start"])
    max_chars = int(cfg.llm.chunk_tokens) * 3
    chunks = chunk_by_patient(turns, max_chars)
    mt = cfg.meeting_types.get(meeting_type, {}).get("ro") or "nespecificat — se deduce din conținut"
    type_choices = "; ".join(f"{k} = {v.get('ro', k)}{TYPE_HINTS.get(k, '')}" for k, v in cfg.meeting_types.items())
    common = dict(meeting_type=mt, date=meeting_date.isoformat(), weekday=WEEKDAYS_RO[meeting_date.weekday()],
                  duration=fmt_ts(duration_s), speakers=", ".join(speakers) or "necunoscut")
    system = SYSTEM.format(out_lang=LANG_NAMES[out_lang])
    stats = {"model": cfg.llm.model, "chunks": len(chunks), "calls": [], "turns": len(turns)}
    t0 = time.time()

    parts = []
    for i, (bed, ch) in enumerate(chunks):
        if progress:
            progress(i / (len(chunks) + 1), f"LLM: partea {i + 1}/{len(chunks)}")
        focus = FOCUS_BED.format(bed=bed) if bed else ""
        part, st = llm.chat_json(system, MAP_PROMPT.format(
            transcript=render_transcript(ch), part=i + 1, parts=len(chunks), focus=focus, **common), MAP_SCHEMA)
        stats["calls"].append(st)
        for tp in part.get("topics", []):
            tp["part"] = i  # rezumatul păstrează o singură situație per pacient și per bucată
        for key in ("decisions", "action_items", "verification_notes"):
            part[key] = [x for x in part.get(key, []) if verify_evidence(x, ch)[0] >= 55]
        parts.append(part)

    mom = merge_parts(parts)
    if progress:
        progress(len(chunks) / (len(chunks) + 1), "LLM: sinteza procesului-verbal")
    compact = {
        "topics": [{"title": t.get("title"), "subject": t.get("subject"), "summary": t.get("summary"),
                    "timestamp": t.get("timestamp")} for t in mom["topics"]],
        "decisions": [f"[{d.get('subject') or '-'}] {d.get('decision')}" for d in mom["decisions"]],
        "action_items": [f"[{x.get('subject') or '-'}] {x.get('task')} — {x.get('owner') or '?'} — "
                         f"{x.get('deadline_text') or 'fără termen'}" for x in mom["action_items"]],
        "open_issues": mom["open_issues"],
        "subjects": sorted({x.get("subject") for k in ("topics", "decisions", "action_items") for x in mom[k]
                            if x.get("subject")}),
    }
    schema = json.loads(json.dumps(SUMMARY_SCHEMA))
    schema["properties"]["meeting_type"] = {"type": "string", "enum": list(cfg.meeting_types)}
    summ, st = llm.chat_json(system, SUMMARY_PROMPT.format(
        extracts=json.dumps(compact, ensure_ascii=False, indent=0), out_lang=LANG_NAMES[out_lang],
        type_choices=type_choices, **common), schema)
    stats["calls"].append(st)
    # subiectele rămân cele extrase direct din transcriere (etapa map): o a doua rescriere pierde valori și doze
    mom.update({k: summ.get(k) for k in ("title", "summary", "meeting_type", "open_issues", "next_meeting")})

    mom = postprocess(mom, turns, speakers, meeting_date)
    mom["language"] = out_lang
    stats["seconds"] = round(time.time() - t0, 1)
    if progress:
        progress(1.0, "LLM: gata")
    return mom, stats


def translate_mom(mom: dict, lang: str) -> dict:
    """Traducerea procesului-verbal (local) — structura și câmpurile verificate rămân neschimbate."""
    if lang == mom.get("language"):
        return mom
    llm = Ollama()
    keep = ("evidence_quote", "evidence_score", "verified", "timestamp", "source_speaker", "source_langs",
            "deadline_date", "id", "owner_speaker", "priority", "speaker", "uncertain_terms", "subject_key",
            "subject_kind", "part")
    payload = {k: mom[k] for k in MOM_SCHEMA["properties"] if k in mom}
    system = f"Ești traducător medical profesionist (română, rusă, engleză). Răspunzi doar în limba {LANG_NAMES[lang]}."
    out, _ = llm.chat_json(system, TRANSLATE_PROMPT.format(out_lang=LANG_NAMES[lang],
                                                           payload=json.dumps(payload, ensure_ascii=False)), MOM_SCHEMA)
    # re-aplicăm câmpurile care nu trebuie să se schimbe la traducere (după poziție)
    for key in ("participants", "topics", "decisions", "action_items", "verification_notes"):
        src, dst = mom.get(key, []), out.get(key, [])
        if len(src) != len(dst):
            out[key] = dst = [dict(s) for s in src] if not dst else dst
        for s, d in zip(src, dst):
            for k in keep:
                if k in s:
                    d[k] = s[k]
    out["language"] = lang
    return out
