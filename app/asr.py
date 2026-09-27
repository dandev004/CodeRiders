"""ASR hibrid pentru code-switching RO / RU / EN (dialect moldovenesc).

De ce nu Whisper „simplu”: Whisper alege o singură limbă per fereastră de 30 s și, pe accentul
moldovenesc, detectorul lui de limbă confundă frecvent româna cu rusa (măsurat pe înregistrarea
Medpark: „Pacientul de pe patul patru” -> ru=0.82 -> „Пациент, патру”). Rezultatul: tot transcriptul
forțat în rusă + bucle de halucinații.

Abordarea Secure MOM — „decode-and-score” la nivel de frază:
  1. VAD (Silero) taie audio în fraze scurte (<= 12 s) — granița naturală a comutărilor de limbă.
  2. Encoder-ul Whisper rulează O SINGURĂ dată per frază (batch pe GPU).
  3. Detecția de limbă e restrânsă la {ro, ru, en} și tratată doar ca indiciu (pe accent moldovenesc
     spune „rusă 96%” pentru fraze clar românești).
  4. Fraza e decodată în limba implicită (ro). Dacă ipoteza e slabă (cuvinte inexistente, scor acustic
     mic, buclă), o decodăm și în limbile candidate. Alegem ipoteza cu scorul
        log-prob medie + w · (proporția de cuvinte reale ale limbii) + prior
     Scorul acustic singur NU ajunge: forțat pe rusă, Whisper transcrie româna drept „rusă fonetică”
     cu log-prob mai bună; testul lexical (app/lexicon.py) o demască.
  5. Filtre anti-halucinație: bucle de repetiție (compression ratio), fraze-fantomă clasice
     („Vă mulțumim pentru vizionare”, „Продолжение следует...”), copierea promptului, no-speech.
  6. Corectarea deterministă a termenilor medicali prin glosar (fuzzy), cu jurnal de audit.
"""
from __future__ import annotations

import gc
import logging
import platform
import re
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import get_config, resolve
from .glossary import get_glossary

log = logging.getLogger("secure_mom.asr")

SR = 16000

# Fraze pe care Whisper le „inventează” pe tăcere/zgomot (învățate din subtitrări YouTube).
HALLUCINATIONS = [
    r"mul[țţ]umi\w* pentru vizionare", r"abona[țţ]i-v[ăa]", r"subtitr\w*", r"nu uita[țţ]i s[ăa] v[ăa] abona[țţ]i",
    r"продолжение следует", r"субтитры", r"спасибо за просмотр", r"подписывайтесь", r"редактор субтитров",
    r"thanks? (you )?for watching", r"please subscribe", r"subtitles by", r"amara\.org",
]
_HALL = re.compile("|".join(HALLUCINATIONS), re.IGNORECASE)
# fraze-fantomă scurte care apar pe tuse / râs / zgomot când întreaga frază e doar asta
_HALL_FULL = re.compile(r"^\W*(и т\.? ?д\.?|и прочее\.?|и так далее\.?|and so on\.?|i'?m just kidding\.?|"
                        r"s[ăa] v[ăa] mul[țţ]umim!?|v[ăa] mul[țţ]umim!?|спасибо\.?|thank you\.?|"
                        r"(ha)+h?|(mm-?)?hmm+|mh-hm)\W*$", re.IGNORECASE)

# Româna moldovenească scrisă cu chirilice (ce produce Whisper când e forțat pe „ru” pe o frază românească)
_MD_CYR = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "j", "з": "z", "и": "i", "й": "i",
    "к": "c", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ț", "ч": "ci", "ш": "ș", "щ": "șt", "ы": "î", "э": "e", "ю": "iu", "я": "ia",
    "ь": "", "ъ": "", "ё": "io",
})
RO_FUNCTION = set("""de la cu pe și si din în in care ce nu da am ai are e a este sunt el ea noi voi ei ele
un o unui unei să sa mai deci acum dar sau pentru după dupa până pana ca că foarte fost face îi ii le lui ei
dânsul dansul dânsa pacientul pacienta patul azi mâine maine bine așa asa tot toate""".split())
RU_FUNCTION = set("""и в не на что я он она с это как а мы у нас есть по но вы его ее её к за то так же
бы уже или да нет был была были мне тоже там здесь все всё только вот ну если когда""".split())


def moldovan_cyrillic_ratio(text: str) -> float:
    """Proporția de cuvinte care, transliterate, sunt cuvinte funcționale românești (vs. rusești)."""
    words = re.findall(r"[а-яё]+", text.lower())
    if len(words) < 3:
        return 0.0
    ro = sum(1 for w in words if w.translate(_MD_CYR) in RO_FUNCTION)
    ru = sum(1 for w in words if w in RU_FUNCTION)
    return ro / max(1, ro + ru)


_CYR = re.compile(r"[а-яёА-ЯЁ]")


@dataclass
class Hyp:
    lang: str
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    temperature: float = 0.0
    words: list | None = None  # [(cuvânt, probabilitate)] — încrederea modelului pe fiecare cuvânt

    def lexical(self) -> float:
        from .lexicon import in_lexicon_rate

        rate, n = in_lexicon_rate(self.text, self.lang)
        # pe 1-2 cuvinte proporția e zgomotoasă — o apropiem de neutru (0.5)
        k = min(1.0, n / 4)
        return k * rate + (1 - k) * 0.5

    def score(self, prior: float, cr_max: float, lex_w: float = 2.0) -> float:
        s = self.avg_logprob + prior + lex_w * self.lexical()
        if self.compression_ratio > cr_max:
            s -= 2.0
        if _HALL.search(self.text):
            s -= 2.0
        if self.lang == "ru" and moldovan_cyrillic_ratio(self.text) > 0.5:
            s -= 1.0  # e română transcrisă cu litere rusești
        if self.lang != "ru" and re.search(r"[а-яёА-ЯЁ]", self.text):
            s -= 1.0  # ipoteză „română” cu litere chirilice = rusă transliterată pe jumătate
        return s


@dataclass
class Segment:
    start: float
    end: float
    lang: str
    text: str
    raw_text: str
    confidence: float
    lang_probs: dict
    alternatives: dict = field(default_factory=dict)
    corrections: list = field(default_factory=list)
    speaker: str | None = None
    flags: list = field(default_factory=list)
    word_conf: list = field(default_factory=list)  # [[cuvânt din text, probabilitate | None]]

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2), "end": round(self.end, 2), "lang": self.lang, "speaker": self.speaker,
            "text": self.text, "raw_text": self.raw_text, "confidence": round(self.confidence, 3),
            "lang_probs": {k: round(v, 3) for k, v in self.lang_probs.items()},
            "corrections": self.corrections, "flags": self.flags, "word_conf": self.word_conf,
        }


def compression_ratio(text: str) -> float:
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


def collapse_repeats(text: str, max_rep: int = 2) -> str:
    """„de cază, de cază, de cază ...” -> „de cază, de cază”. Taie buclele de n-grame (1..8 cuvinte)."""
    words = text.split()
    changed = True
    while changed:
        changed = False
        for n in range(1, 9):
            i = 0
            out: list[str] = []
            while i < len(words):
                gram = words[i:i + n]
                reps = 1
                while words[i + reps * n:i + (reps + 1) * n] == gram and len(gram) == n:
                    reps += 1
                if reps > max_rep:
                    out += gram * max_rep
                    i += reps * n
                    changed = True
                else:
                    out.append(words[i])
                    i += 1
            words = out
    text = " ".join(words)
    # repetiții de silabe: „е-е-е-е-е-е”
    return re.sub(r"(\b\w{1,3}-)\1{3,}", r"\1\1", text)


def _prompt_leak(text: str, prompt: str) -> bool:
    """Pe audio neclar Whisper recită promptul — detectăm și aruncăm."""
    if not prompt or not text:
        return False
    tw = set(re.findall(r"\w+", text.lower()))
    pw = set(re.findall(r"\w+", prompt.lower()))
    return len(tw) >= 2 and len(tw & pw) / len(tw) > 0.7


# ----------------------------------------------------------------------------------------------
# Backend-uri
# ----------------------------------------------------------------------------------------------

class MLXBackend:
    """Apple Silicon (Metal GPU) — pentru laptopurile de demo."""

    name = "mlx"

    def __init__(self, model_path: str):
        import mlx.core as mx
        from mlx_whisper.load_models import load_model
        from mlx_whisper.tokenizer import get_tokenizer

        self.mx = mx
        self.model_path = model_path
        # fără limită, cache-ul Metal reține buffere pentru fiecare formă de batch și crește până la swap
        # (măsurat: 11.5 GB „wired” + 12 GB swap pe un Mac de 16 GB). Limităm cache-ul și îl golim periodic.
        limit = int(get_config().asr.get("mlx_cache_limit_gb", 1) * 1024 ** 3)
        try:
            mx.set_cache_limit(limit)
        except AttributeError:
            mx.metal.set_cache_limit(limit)
        self.model = load_model(model_path, dtype=mx.float16)
        self.tokenizer = get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages)

    def encode(self, segs: list[np.ndarray]):
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim

        mx = self.mx
        mels = [log_mel_spectrogram(pad_or_trim(mx.array(s), N_SAMPLES), n_mels=self.model.dims.n_mels) for s in segs]
        feats = self.model.encoder(mx.stack(mels).astype(mx.float16))
        mx.eval(feats)
        return feats

    def lang_probs(self, feats, langs: list[str]) -> list[dict]:
        from mlx_whisper.decoding import detect_language

        _, probs = detect_language(self.model, feats, self.tokenizer)
        out = []
        for p in probs:
            sub = {l: float(p.get(l, 0.0)) for l in langs}
            z = sum(sub.values()) or 1.0
            out.append({l: v / z for l, v in sub.items()})
        return out

    def decode(self, feats, lang: str, prompt: str | None, temperature: float, max_tokens: int = 224) -> list[Hyp]:
        from mlx_whisper.decoding import DecodingOptions, decode

        opts = DecodingOptions(language=lang, task="transcribe", temperature=temperature, prompt=prompt or None,
                               without_timestamps=True, fp16=True, sample_len=max_tokens)
        res = decode(self.model, feats, opts)
        res = res if isinstance(res, list) else [res]
        words = self.word_probs(feats, [r.tokens for r in res], lang, prompt)
        return [Hyp(lang, r.text.strip(), float(r.avg_logprob), float(r.no_speech_prob),
                    compression_ratio(r.text), temperature, w) for r, w in zip(res, words)]

    def word_probs(self, feats, token_lists: list[list[int]], lang: str, prompt: str | None) -> list[list]:
        """Probabilitatea fiecărui cuvânt: o singură trecere a decoderului peste tokenii deja aleși (teacher forcing),
        cu același prefix ca la decodare. Cuvântul are probabilitatea celui mai nesigur token al său."""
        from mlx_whisper.tokenizer import get_tokenizer

        mx = self.mx
        tok = get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages, language=lang,
                            task="transcribe")
        pre = list(tok.sot_sequence_including_notimestamps)
        if prompt:
            pre = [tok.sot_prev] + tok.encode(" " + prompt.strip())[-(self.model.dims.n_text_ctx // 2 - 1):] + pre
        L = max(1, max((len(t) for t in token_lists), default=1))
        inp = np.full((len(token_lists), len(pre) + L), tok.eot, dtype=np.int32)
        for i, t in enumerate(token_lists):
            inp[i, :len(pre) + len(t)] = pre + list(t)
        logits = self.model.decoder(mx.array(inp), feats)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(lp)
        lp = np.array(lp)
        out = []
        for i, t in enumerate(token_lists):
            words: list = []
            for k, tid in enumerate(t):
                if tid >= tok.eot:
                    continue
                piece = tok.decode([tid])
                p = float(np.exp(lp[i, len(pre) + k - 1, tid]))
                if not words or piece.startswith(" ") or not piece.strip():
                    words.append([piece, p])
                else:
                    words[-1][0] += piece
                    words[-1][1] = min(words[-1][1], p)
            out.append([(w.strip(), round(p, 3)) for w, p in words if w.strip()])
        return out

    def select(self, feats, idx: list[int]):
        return feats[self.mx.array(idx)]

    def clear_cache(self):
        try:
            self.mx.clear_cache()
        except AttributeError:
            self.mx.metal.clear_cache()

    def release(self):
        self.model = None
        gc.collect()
        try:
            self.mx.clear_cache()
        except AttributeError:
            self.mx.metal.clear_cache()


class FasterWhisperBackend:
    """CTranslate2 — serverul spitalului (GPU 16 GB fp16 sau CPU int8)."""

    name = "faster-whisper"

    def __init__(self, model_path: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        if device == "auto":
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        self.model_path = model_path
        self.model = WhisperModel(model_path, device=device, compute_type=compute_type)
        log.info("faster-whisper on %s/%s", device, compute_type)

    # faster-whisper nu expune encoder-ul la fel de comod — păstrăm audio și decodăm per segment.
    def encode(self, segs):
        return list(segs)

    def lang_probs(self, feats, langs):
        out = []
        for s in feats:
            _, _, allp = self.model.detect_language(audio=s, language_detection_segments=1)
            d = {l: 0.0 for l in langs}
            for code, p in allp:
                if code in d:
                    d[code] = float(p)
            z = sum(d.values()) or 1.0
            out.append({l: v / z for l, v in d.items()})
        return out

    def decode(self, feats, lang, prompt, temperature, max_tokens: int = 224):
        hyps = []
        for s in feats:
            segs, _ = self.model.transcribe(
                s, language=lang, initial_prompt=prompt or None, temperature=temperature,
                beam_size=int(get_config().asr.get("beam_size", 1)) if temperature == 0 else 1,
                condition_on_previous_text=False, without_timestamps=True, vad_filter=False, max_new_tokens=max_tokens,
                compression_ratio_threshold=None, log_prob_threshold=None, no_speech_threshold=None,
                word_timestamps=True)
            segs = list(segs)
            words = [(w.word.strip(), round(float(w.probability), 3)) for x in segs for w in (x.words or [])]
            text = " ".join(x.text.strip() for x in segs).strip()
            lp = float(np.mean([x.avg_logprob for x in segs])) if segs else -5.0
            ns = float(np.mean([x.no_speech_prob for x in segs])) if segs else 1.0
            hyps.append(Hyp(lang, text, lp, ns, compression_ratio(text), temperature, words or None))
        return hyps

    def select(self, feats, idx):
        return [feats[i] for i in idx]

    def clear_cache(self):
        pass

    def release(self):
        self.model = None
        gc.collect()


def _profile_cfg(profile: str | None):
    cfg = get_config().asr
    prof = (cfg.get("profiles") or {}).get(profile or cfg.get("profile", ""))
    return type(cfg)({**cfg, **prof}) if prof else cfg


def _backend_kind(cfg) -> str:
    if cfg.backend != "auto":
        return cfg.backend
    return "mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "faster-whisper"


def load_backend(profile: str | None = None):
    """Modelul multilingv: detecția de limbă + rusă/engleză (și română, dacă nu există specialist)."""
    cfg = _profile_cfg(profile)
    if _backend_kind(cfg) == "mlx":
        return MLXBackend(str(resolve(cfg.mlx_model)))
    return FasterWhisperBackend(str(resolve(cfg.faster_whisper_model)), cfg.device, cfg.compute_type)


def load_specialist(profile: str | None = None):
    """Modelul specialist pentru limba de stat (Whisper adaptat pe româna moldovenească), dacă e configurat."""
    cfg = _profile_cfg(profile)
    key = "ro_mlx_model" if _backend_kind(cfg) == "mlx" else "ro_faster_whisper_model"
    # o listă = ordine de preferință: modelul adaptat medical, apoi cel moldovenesc de bază (dacă nu s-a antrenat)
    paths = cfg.get(key) or []
    path = next((p for p in ([paths] if isinstance(paths, str) else paths) if resolve(p).exists()), None)
    if not path:
        return None
    if _backend_kind(cfg) == "mlx":
        return MLXBackend(str(resolve(path)))
    return FasterWhisperBackend(str(resolve(path)), cfg.device, cfg.compute_type)


# ----------------------------------------------------------------------------------------------
# VAD + transcriere
# ----------------------------------------------------------------------------------------------

def vad_segments(audio: np.ndarray) -> list[tuple[int, int]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    v = get_config().vad
    ts = get_speech_timestamps(audio, VadOptions(
        threshold=v.threshold, min_silence_duration_ms=v.min_silence_ms, max_speech_duration_s=v.max_segment_s,
        speech_pad_ms=v.speech_pad_ms, min_speech_duration_ms=v.min_speech_ms))
    return [(t["start"], t["end"]) for t in ts if t["end"] - t["start"] >= 0.3 * SR]


def transcribe(audio: np.ndarray, meeting_type: str | None = None,
               progress: Callable[[float, str], None] | None = None,
               backend=None, batch_size: int = 8, vad_audio: np.ndarray | None = None,
               profile: str | None = None) -> tuple[list[Segment], dict]:
    # VAD pe audio curățat (zgomotul nu mai e confundat cu vorbire), decodare pe audio „light”
    spans = vad_segments(vad_audio if vad_audio is not None else audio)
    backend = backend or load_backend(profile)
    specialist = load_specialist(profile)
    try:
        segs, st = decode_spans(audio, spans, meeting_type, progress, backend, batch_size, specialist=specialist)
    finally:
        backend.release()
        if specialist:
            specialist.release()
    st["profile"] = profile or get_config().asr.get("profile", "accurate")
    st["model"] = getattr(backend, "model_path", st.get("model"))
    st["ro_model"] = getattr(specialist, "model_path", None)
    return segs, st


def decode_spans(audio: np.ndarray, spans: list[tuple[int, int]], meeting_type: str | None = None,
                 progress: Callable[[float, str], None] | None = None, backend=None, batch_size: int = 8,
                 languages: list[str] | None = None, specialist=None) -> tuple[list[Segment], dict]:
    """specialist = model pentru limba de stat (RO moldovenesc); backend = model multilingv (LID, RU, EN).

    Măsurat (training/eval_codeswitch.py): modelul adaptat pe dialect „uită” rusa (o scrie cu litere românești),
    iar large-v3 forțat pe română TRADUCE rusa și engleza. De aceea: româna o transcrie specialistul, rusa și
    engleza modelul multilingv, iar testul lexical decide care ipoteză e un text real în limba ei."""
    cfg = get_config().asr
    langs = list(languages or cfg.languages)
    default = cfg.default_language
    prior = dict(cfg.language_prior)
    cr_max = float(cfg.compression_ratio_max)
    lex_w = float(cfg.get("lexicon_weight", 2.0))
    acc_lex = float(cfg.get("accept_if_lexicon_above", 0.72))
    acc_lp = float(cfg.get("accept_if_logprob_above", -0.8))
    min_wps = float(cfg.get("min_words_per_second", 1.0))
    tps = float(cfg.get("tokens_per_second", 9))
    temps = [float(t) for t in cfg.temperatures]
    gl = get_glossary()
    fuzzy_thr = int(get_config().glossary.fuzzy_threshold)
    prompts = {l: (gl.whisper_prompt(l, meeting_type) if cfg.use_glossary_prompt else None) for l in langs}

    def score(h: Hyp) -> float:
        return h.score(prior.get(h.lang, 0.0), cr_max, lex_w)

    def n_words(h: Hyp) -> int:
        return len(re.findall(r"\w+", h.text))

    def truncated(h: Hyp, dur: float) -> bool:
        # modelul specialist, antrenat doar pe română, se poate opri când fraza trece în altă limbă
        # („Facem ecocardiografie și...” fără partea în engleză) — prea puține cuvinte pentru durata audio
        return dur >= 2.5 and n_words(h) / dur < min_wps

    t0 = time.time()
    # frazele de lungimi apropiate în același batch -> limita de tokeni strânsă, fără timp pierdut pe padding
    order = sorted(range(len(spans)), key=lambda i: spans[i][1] - spans[i][0])
    own_backend = backend is None
    backend = backend or load_backend()
    stats = {"backend": backend.name, "model": str(cfg.mlx_model if backend.name == "mlx" else cfg.faster_whisper_model),
             "vad_segments": len(spans), "second_pass": 0, "dropped": 0, "fallback_temperature": 0,
             "lang_seconds": {l: 0.0 for l in langs}}

    results: dict[int, Segment] = {}
    for b0 in range(0, len(order), batch_size):
        ids = order[b0:b0 + batch_size]
        batch = [spans[i] for i in ids]
        max_tok = min(224, int(tps * max(e - s for s, e in batch) / SR) + 12)
        clips = [audio[s:e] for s, e in batch]
        feats = backend.encode(clips)
        probs = backend.lang_probs(feats, langs)
        sfeats = specialist.encode(clips) if specialist else feats
        sbe = specialist or backend

        def be_for(lang):
            return (sbe, sfeats) if lang == default else (backend, feats)

        # 1) toată lumea în limba implicită (modelul specialist, dacă există)
        best: list[Hyp] = sbe.decode(sfeats, default, prompts[default], 0.0, max_tok)
        tried = [{default: h} for h in best]

        # 2) a doua trecere doar pentru ipotezele slabe, grupate pe limbă (batch)
        need: dict[str, list[int]] = {}
        short: list[int] = []
        for i, (h, p) in enumerate(zip(best, probs)):
            dur = (batch[i][1] - batch[i][0]) / SR
            cut = truncated(h, dur)
            good = h.lexical() >= acc_lex and h.avg_logprob >= acc_lp and h.compression_ratio <= cr_max \
                and not _HALL.search(h.text) and not _prompt_leak(h.text, prompts[default]) \
                and not _CYR.search(h.text) and not cut
            if good:
                continue
            if cut and specialist:
                short.append(i)
            for l in langs:
                if l != default and (p[l] >= cfg.second_pass_min_prob or cut):
                    need.setdefault(l, []).append(i)
        if short:  # a doua opinie în limba implicită, de la modelul multilingv (nu se oprește la schimbarea de limbă)
            stats["second_pass"] += len(short)
            for i, h in zip(short, backend.decode(backend.select(feats, short), default, prompts[default], 0.0, max_tok)):
                tried[i][default + "*"] = h
        for l, idx in need.items():
            stats["second_pass"] += len(idx)
            for i, h in zip(idx, backend.decode(backend.select(feats, idx), l, prompts[l], 0.0, max_tok)):
                tried[i][l] = h

        # 3) alegere + fallback de temperatură (fără prompt) pentru bucle / prompt recitat / fraze-fantomă
        for i, (s, e) in enumerate(batch):
            cands = tried[i]
            most = max(n_words(h) for h in cands.values())
            # ipoteza care acoperă mult mai puțin din vorbire decât alta e, cel mai probabil, trunchiată
            win = max(cands.values(), key=lambda h: score(h) - (1.5 if most >= 4 and n_words(h) < 0.5 * most else 0.0))
            if win.compression_ratio > cr_max or _prompt_leak(win.text, prompts[win.lang]) or _HALL.search(win.text):
                for t in temps[1:]:
                    stats["fallback_temperature"] += 1
                    be, fe = be_for(win.lang)
                    h = be.decode(be.select(fe, [i]), win.lang, None, t, max_tok)[0]
                    if h.compression_ratio <= cr_max and not _HALL.search(h.text):
                        win = h
                        break
            seg = _finalize(win, s, e, probs[i], cands, prompts, gl, fuzzy_thr, cr_max)
            if seg is None:
                stats["dropped"] += 1
                continue
            stats["lang_seconds"][seg.lang] += seg.end - seg.start
            results[ids[i]] = seg
        del feats, sfeats
        backend.clear_cache()
        if specialist:
            specialist.clear_cache()
        if progress:
            progress(min(1.0, (b0 + len(batch)) / max(1, len(spans))), f"{len(results)} fraze transcrise")

    if own_backend:
        backend.release()
    stats["seconds"] = round(time.time() - t0, 1)
    stats["lang_seconds"] = {k: round(v, 1) for k, v in stats["lang_seconds"].items()}
    return [results[i] for i in sorted(results)], stats


def _finalize(h: Hyp, s: int, e: int, probs: dict, cands: dict, prompts: dict, gl, fuzzy_thr: int,
              cr_max: float) -> Segment | None:
    text = h.text.strip()
    flags = []
    if not text or _HALL.search(text) and len(text.split()) <= 8 or _HALL_FULL.match(text):
        return None
    # frază scurtă, în altă limbă decât cea implicită, cu încredere mică = de regulă zgomot „tradus”
    if h.lang != get_config().asr.default_language and len(text.split()) <= 3 and h.avg_logprob < -0.75:
        return None
    if _prompt_leak(text, prompts.get(h.lang) or ""):
        return None
    if h.no_speech_prob > 0.6 and h.avg_logprob < -1.0:
        return None
    if h.compression_ratio > cr_max:
        text = collapse_repeats(text)
        flags.append("repetition_removed")
    if h.avg_logprob < -1.1:
        flags.append("low_confidence")
    if h.lang == "ro":
        from .ronum import to_digits
        text = to_digits(text)
    text = text[:1].upper() + text[1:]
    raw = text
    text, changes = gl.correct(text, h.lang, fuzzy_thr)
    # termeni medicali deformați (vocabular SiMoNERo/MoNERo + glosar), doar pentru cuvinte inexistente
    from .medcorrect import phonetic_fix
    text, ph = phonetic_fix(text, h.lang, int(get_config().medcorrect.get("phonetic_threshold", 92)))
    changes = list(changes) + [(c["from"], c["to"]) for c in ph]
    from .medcorrect import is_critical
    lo = float(get_config().asr.get("word_confidence_low", 0.5))
    wc = [[w, p, bool(p is not None and p < lo and is_critical(w, h.lang))] for w, p in align_word_conf(h.words, text)]
    if any(p is not None and p < lo for _, p, _ in wc):
        flags.append("uncertain_words")
    return Segment(
        start=s / SR, end=e / SR, lang=h.lang, text=text, raw_text=raw,
        confidence=float(np.exp(h.avg_logprob)), lang_probs=probs,
        alternatives={l: {"text": c.text, "avg_logprob": round(c.avg_logprob, 3)} for l, c in cands.items() if l != h.lang},
        corrections=[{"from": a, "to": b} for a, b in changes], flags=flags, word_conf=wc,
    )


def align_word_conf(raw_words: list | None, final: str) -> list:
    """Duce probabilitățile cuvintelor ASR peste textul final (numerale în cifre, termeni corectați).

    Aliniere pe cuvinte; un cuvânt final care înlocuiește mai multe cuvinte ASR („patru opt” -> „patul 8”)
    primește probabilitatea cea mai mică dintre ele — o corectură nu face un cuvânt nesigur mai sigur."""
    import difflib

    fw = final.split()
    if not raw_words:
        return [[w, None] for w in fw]
    key = lambda w: re.sub(r"[^\w]", "", w.lower())  # noqa: E731
    a, b = [key(w) for w, _ in raw_words], [key(w) for w in fw]
    out: list = [[w, None] for w in fw]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            for k in range(j2 - j1):
                out[j1 + k][1] = raw_words[i1 + k][1]
        elif op == "replace" and i2 > i1:
            p = min(pr for _, pr in raw_words[i1:i2])
            for k in range(j1, j2):
                out[k][1] = p
    return out
