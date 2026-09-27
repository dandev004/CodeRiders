"""Formatare clinică a procesului-verbal: abrevieri medicale standard și evidențierea valorilor anormale.

Totul e determinist (reguli + intervale de referință), aplicat doar la afișare — JSON-ul procesului-verbal rămâne
textul LLM-ului, neatins. Nu inventăm nimic: evidențiem doar ce e scris, iar „anormal” înseamnă în afara
intervalului de referință de mai jos (adult, unități uzuale în Republica Moldova).
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------------------------------------
# Abrevieri
# ------------------------------------------------------------------------------------------------

_I = re.IGNORECASE
ABBR = {
    "ro": [
        (r"\b(?:tensiune|tensiunea|presiune|presiunea)(?:a)? arterial[ăae]\b", "TA"),
        (r"\btensiun(?:e|ea|ile)(?= (?:de |la |a fost |este |e )?\d)", "TA"),
        (r"\bsatura(?:ț|t)i(?:a|e|ei)(?: periferică)?(?: (?:cu|în|de) oxigen| O2| cu O2)?\b", "SpO2"),
        (r"\bnoradrenalin(?:ă|a|ei)\b", "NA"),
        (r"\bfrac(?:ț|t)i(?:a|e|ei) de ejec(?:ț|t)ie(?: a ventriculului st[âa]ng| VS)?\b", "FEVS"),
        (r"\bfrecven(?:ț|t)(?:a|ă|ei) cardiac[ăae]\b|\balur(?:a|ă) ventricular[ăa]\b", "AV"),
        (r"\bfrecven(?:ț|t)(?:a|ă|ei) respiratori[ei]\b", "FR"),
        (r"\belectrocardiogram(?:a|ă|ei)\b", "ECG"),
        (r"\becocardiografi(?:a|e|ei)\b", "EcoCG"),
        (r"\btomografi(?:a|e|ei) computerizat[ăae]\b", "CT"),
        (r"\brezonan(?:ț|t)(?:a|ă|ei) magnetic[ăae](?: nuclear[ăa])?\b", "IRM"),
        (r"\bhemoglobin(?:a|ă|ei)\b", "Hb"),
        (r"\binfarct(?:ul)? miocardic acut\b", "IMA"),
        (r"\binfarct(?:ul)? miocardic\b", "IM"),
        (r"\binsuficien(?:ț|t)(?:a|ă) renal[ăa] acut[ăa]\b", "IRA"),
        (r"\binsuficien(?:ț|t)(?:a|ă) cardiac[ăa]\b", "IC"),
        (r"\bproteina C reactiv[ăa]\b", "PCR"),
        (r"\bterapie intensiv[ăa]\b|\bterapia intensiv[ăa]\b", "ATI"),
        (r"\bintravenos\b", "i.v."),
        (r"\bsubcutanat\b", "s.c."),
    ],
    "en": [
        (r"\bblood pressure\b", "BP"),
        (r"\boxygen saturation\b|\bsaturation\b", "SpO2"),
        (r"\bnorepinephrine\b|\bnoradrenaline\b", "NE"),
        (r"\b(?:left ventricular )?ejection fraction\b", "LVEF"),
        (r"\bheart rate\b", "HR"),
        (r"\brespiratory rate\b", "RR"),
        (r"\belectrocardiogram\b", "ECG"),
        (r"\bechocardiograph(?:y|ic)\b", "echo"),
        (r"\bcomputed tomography\b", "CT"),
        (r"\bha?emoglobin\b", "Hb"),
        (r"\bintensive care unit\b|\bintensive care\b", "ICU"),
        (r"\bintravenous(?:ly)?\b", "IV"),
        (r"\bC-reactive protein\b", "CRP"),
        (r"\bacute kidney injury\b", "AKI"),
        (r"\bmyocardial infarction\b", "MI"),
    ],
    "ru": [
        (r"\bартериальн\w* давлени\w*\b", "АД"),
        (r"\bсатураци\w*\b", "SpO2"),
        (r"\bноррадреналин\w*\b|\bнорадреналин\w*\b", "НА"),
        (r"\bфракци\w* выброса(?: левого желудочка)?\b", "ФВ ЛЖ"),
        (r"\bчастот\w* сердечных сокращений\b", "ЧСС"),
        (r"\bчастот\w* дыхани\w*\b|\bчастот\w* дыхательных движений\b", "ЧДД"),
        (r"\bэлектрокардиограмм\w*\b", "ЭКГ"),
        (r"\bкомпьютерн\w* томографи\w*\b", "КТ"),
        (r"\bгемоглобин\w*\b", "Hb"),
        (r"\bвнутривенно\b", "в/в"),
        (r"\bинфаркт\w* миокарда\b", "ИМ"),
    ],
}
_ABBR = {k: [(re.compile(p, _I), r) for p, r in v] for k, v in ABBR.items()}


def abbreviate(text: str | None, lang: str) -> str:
    if not text:
        return text or ""
    for rx, rep in _ABBR.get(lang, []):
        text = rx.sub(rep, text)
    return text


# ------------------------------------------------------------------------------------------------
# Valori anormale și doze
# ------------------------------------------------------------------------------------------------

def _f(x: str) -> float:
    return float(x.replace(",", "."))


# (denumiri, funcție de verificare: valoare -> e anormală?)
LABS: list[tuple[str, callable]] = [
    (r"Hb|hemoglobin\w*|гемоглобин\w*|ha?emoglobin",
     lambda v: (v < 110 or v > 180) if v > 25 else (v < 11 or v > 18)),         # g/L sau g/dL
    (r"creatinin\w*|креатинин\w*", lambda v: v > 110 if v > 20 else v > 1.3),  # µmol/L sau mg/dL
    (r"lactat\w*|лактат\w*|lactate", lambda v: v > 2.0),
    (r"potasiu\w*|kaliemi\w*|калий\w*|potassium|K\+?", lambda v: v < 3.5 or v > 5.3),
    (r"sodiu\w*|natremi\w*|натрий\w*|sodium|Na\+", lambda v: v < 135 or v > 145),
    (r"glicemi\w*|glucoz\w*|глюкоз\w*|glucose", lambda v: (v < 3.9 or v > 10) if v < 30 else (v < 70 or v > 180)),
    (r"leucocit\w*|лейкоцит\w*|WBC|leukocytes", lambda v: v < 4 or v > 11),
    (r"trombocit\w*|тромбоцит\w*|platelets", lambda v: v < 150 or v > 450),
    (r"PCR|CRP|СРБ", lambda v: v > 10),
    (r"procalcitonin\w*|прокальцитонин\w*", lambda v: v > 0.5),
    (r"INR", lambda v: v > 1.3),
    (r"uree\w*|ureea|мочевин\w*|urea", lambda v: v > 8.3),
    (r"bilirubin\w*|билирубин\w*", lambda v: v > 21),
    (r"pH", lambda v: v < 7.35 or v > 7.45),
    (r"pCO2|PaCO2", lambda v: v < 35 or v > 45),
    (r"pO2|PaO2", lambda v: v < 60),
    (r"FEVS|ФВ ЛЖ|LVEF|EF", lambda v: v < 50),
    (r"SpO2|saturați\w*", lambda v: v < 92),
    (r"AV|FC|puls\w*|ЧСС|HR|pulse", lambda v: 20 <= v <= 250 and (v < 50 or v > 110)),   # „bloc AV 2” nu e puls
    (r"FR|ЧДД|RR", lambda v: 4 <= v <= 70 and (v < 10 or v > 24)),
    (r"temperatur\w*|T°|температур\w*", lambda v: 30 <= v <= 45 and (v >= 38 or v < 35.5)),
    (r"diurez\w*|диурез\w*", lambda v: v < 500 if v > 50 else False),
]
_NUM = r"(-?\d+(?:[.,]\d+)?)"
_LAB = [(re.compile(rf"\b(?:{names})\s*(?:de|=|:|la|a scăzut la|a crescut la|scade la|crește la|—|-)?\s*{_NUM}\s*%?", _I), bad)
        for names, bad in LABS]
_BP = re.compile(r"\b(?:TA|АД|BP)\s*(?:de|=|:|la)?\s*(\d{2,3})\s*(?:/|pe|на)\s*(\d{2,3})(?:\s*mm ?Hg)?|"
                 r"\b(\d{2,3})\s*/\s*(\d{2,3})\s*mm ?Hg", _I)
_UNIT = r"(?:µg|mcg|мкг|mg|мг|g|г|ml|мл|mL|UI|U|ЕД|mmol|ммоль|mEq)(?:/(?:kg|кг))?(?:/(?:min|мин|h|oră|ч|zi|сут|day))?"
_DOSE = re.compile(rf"\b\d+(?:[.,]\d+)?\s?{_UNIT}(?![\w/])", _I)
_VASO = re.compile(r"\b(?:(?-i:NA|НА|NE)|dobutamin\w*|Dobu|добутамин\w*|adrenalin\w*|адреналин\w*|epinephrine|"
                   r"dopamin\w*|допамин\w*|дофамин\w*|vasopresin\w*|вазопрессин\w*)\s*(?:în doză de|doza de|de|la|—|-)?\s*"
                   r"\d+(?:[.,]\d+)?(?:\s?" + _UNIT + ")?", _I)


def alert_spans(text: str) -> list[tuple[int, int]]:
    """Intervalele de text de pus în bold: doze, suport vasopresor, semne vitale și analize în afara normei."""
    spans: list[tuple[int, int]] = []
    for m in _BP.finditer(text):
        sys_, dia = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        s, d = int(sys_), int(dia)
        if s < 90 or s >= 160 or d < 60 or d >= 100:
            spans.append(m.span())
    for rx, bad in _LAB:
        for m in rx.finditer(text):
            try:
                if bad(_f(m.group(1))):
                    spans.append(m.span())
            except ValueError:
                continue
    spans += [m.span() for m in _DOSE.finditer(text)]
    spans += [m.span() for m in _VASO.finditer(text)]
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def runs(text: str | None, lang: str) -> list[tuple[str, bool]]:
    """Text clinic (abreviat) împărțit în bucăți (text, bold)."""
    t = abbreviate(text, lang)
    out, pos = [], 0
    for s, e in alert_spans(t):
        if s > pos:
            out.append((t[pos:s], False))
        out.append((t[s:e].strip(), True))
        if t[s:e].endswith(" "):
            out.append((" ", False))
        pos = e
    if pos < len(t):
        out.append((t[pos:], False))
    return out


# ------------------------------------------------------------------------------------------------
# Subiecte: pacient (după pat) sau temă
# ------------------------------------------------------------------------------------------------

_BED = re.compile(r"\b(?:pat(?:ul|ului|urile|urilor)?|salon\w*|bed|кой(?:ка|ке|ки|ку)|кроват\w*)\s*(?:nr\.?|№|number)?\s*"
                  r"(\d{1,3})\b", _I)
_PAT_WORD = re.compile(r"\b(pacient\w*|patient\w*|пациент\w*|больн\w*|bolnav\w*)\b", _I)
BED_LABEL = {"ro": "Pacient — Patul {n}", "ru": "Пациент — койка {n}", "en": "Patient — Bed {n}"}


def bed_of(*texts: str | None) -> str | None:
    for t in texts:
        if t:
            m = _BED.search(t)
            if m:
                return m.group(1)
    return None


def is_patient(text: str | None) -> bool:
    return bool(text and (_PAT_WORD.search(text) or _BED.search(text)))


# ------------------------------------------------------------------------------------------------
# Unități de măsură: rămân doar dacă au fost rostite în ședință
# ------------------------------------------------------------------------------------------------

# (unitatea în textul LLM-ului, formele în care ar apărea în transcriere dacă a fost rostită)
UNITS = [
    (r"mm ?Hg", ["mmhg", "milimetr", "мм рт", "ртутн"]),
    (r"(?:mg|g)/dL", ["decilitr", "/dl", "децилитр"]),
    (r"(?:µ|μ|u)mol/L", ["micromol", "µmol", "umol", "мкмоль"]),
    (r"mmol/L", ["milimol", "mmol", "ммоль"]),
    (r"(?:mcg|µg|μg)/kg/min", ["microgram", "mcg", "gama", "gamma", "мкг", "гамм"]),
    (r"mg/kg(?:/(?:zi|h|oră))?", ["pe kilogram", "mg/kg", "на кило", "мг/кг"]),
    (r"g/L", ["grame pe litru", "g/l", "г/л"]),
    (r"mg/(?:min|h|oră)", ["mg/", "miligram", "мг/"]),
    (r"ml/(?:h|oră|kg)", ["ml/", "mililitr", "мл/"]),
    (r"mg", ["miligram", " mg", "мг", "милиграм"]),
    (r"ml", ["mililitr", " ml", "мл", "миллилитр"]),
    (r"UI", ["unități", "unitati", " ui ", "единиц", " ед"]),
]
_UNITS = [(re.compile(rf"\s*\b{u}(?![\w/])", re.IGNORECASE), spoken) for u, spoken in UNITS]


def strip_unspoken_units(text: str | None, transcript: str) -> str | None:
    """Scoate unitățile adăugate de LLM care nu apar nicăieri în ședință („TA 80/40 mmHg” -> „TA 80/40”)."""
    if not text:
        return text
    low = transcript.lower()
    for rx, spoken in _UNITS:
        if not any(w in low for w in spoken):
            text = rx.sub("", text)
    return text


def bed_timeline(turns: list[dict]) -> list[tuple[float, str]]:
    """(momentul, patul) pentru fiecare pat rostit — la consiliu pacienții se discută pe rând."""
    return [(t["start"], m.group(1)) for t in turns for m in _BED.finditer(t.get("text") or "")]


def bed_at(timeline: list[tuple[float, str]], t: float) -> str | None:
    """Patul discutat la momentul t: ultimul pat rostit până atunci."""
    cur = None
    for ts, bed in timeline:
        if ts > t + 1:
            break
        cur = bed
    return cur
