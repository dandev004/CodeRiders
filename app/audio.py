"""Preprocesare audio: decodare orice format (m4a, mp3, wav, webm, mp4 video...), curățare zgomot, normalizare.

Lanțul de curățare (configurabil în config.yaml -> audio):
  1. highpass  — taie vibrații joase, pași, lovituri în masă, hum de rețea
  2. lowpass   — taie șuieratul de înaltă frecvență (peste banda vocii utile pentru ASR)
  3. dynaudnorm — normalizare dinamică: vorbitorii departe de microfon ajung la nivelul celor apropiați
  4. afftdn     — denoise FFT pentru zgomot de fond staționar (ventilație, aer condiționat, aparatură)
  5. noisereduce (spectral gating non-staționar) — zgomot variabil: foșnet de hârtie, hârșâieli, pocnituri,
     voci din fundal
Un singur apel ffmpeg produce ambele variante (asplit): ~3 s pentru 12 min de audio
(loudnorm + adeclick costau 45 s — loudnorm face upsampling intern la 192 kHz).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import get_config

log = logging.getLogger("secure_mom.audio")


def _ffmpeg_filters(cfg, full: bool = True) -> str:
    a = cfg.audio
    f = []
    if a.get("highpass_hz"):
        f.append(f"highpass=f={a.highpass_hz}")
    if a.get("lowpass_hz"):
        f.append(f"lowpass=f={a.lowpass_hz}")
    if a.get("normalize"):
        f.append("dynaudnorm=f=250:g=15:p=0.9:m=10")
    if full and a.get("denoise"):
        # nf = noise floor; strength 0..1 -> reducere 6..20 dB
        nr = 6 + 14 * float(a.get("denoise_strength", 0.8))
        f.append(f"afftdn=nr={nr:.0f}:nf=-40:tn=1")
    return ",".join(f) or "anull"


def _run_ffmpeg(cmd: list[str], outputs: list[Path]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for dst in outputs:
        if not dst.exists() or dst.stat().st_size < 1000:
            raise RuntimeError(f"ffmpeg nu a putut decoda fișierul audio: {proc.stderr.strip()[:500]}")
    if proc.returncode != 0:
        # ex. ultimul cadru corupt la înregistrări m4a — fișierul rezultat e totuși valid
        log.warning("ffmpeg warnings: %s", proc.stderr.strip()[:300])


def decode(src: Path, dst: Path, clean: bool = True, full: bool = True) -> Path:
    cfg = get_config()
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar",
           str(cfg.audio.sample_rate)]
    if clean:
        cmd += ["-af", _ffmpeg_filters(cfg, full)]
    cmd += ["-c:a", "pcm_s16le", str(dst)]
    _run_ffmpeg(cmd, [dst])
    return dst


def decode_both(src: Path, asr_dst: Path, clean_dst: Path) -> None:
    """O singură decodare -> două ieșiri: varianta ușoară (ASR) și cea curățată complet (VAD/diarizare)."""
    cfg = get_config()
    graph = (f"[0:a]aformat=channel_layouts=mono,aresample={cfg.audio.sample_rate},asplit=2[a][b];"
             f"[a]{_ffmpeg_filters(cfg, False)}[o1];[b]{_ffmpeg_filters(cfg, True)}[o2]")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-filter_complex", graph,
           "-map", "[o1]", "-c:a", "pcm_s16le", str(asr_dst), "-map", "[o2]", "-c:a", "pcm_s16le", str(clean_dst)]
    _run_ffmpeg(cmd, [asr_dst, clean_dst])


def spectral_denoise(audio: np.ndarray, sr: int, strength: float) -> np.ndarray:
    import noisereduce as nr

    out = nr.reduce_noise(
        y=audio, sr=sr, stationary=False, prop_decrease=min(0.9, 0.5 + 0.4 * strength),
        time_constant_s=2.0, freq_mask_smooth_hz=500, time_mask_smooth_ms=50,
        chunk_size=sr * 60, n_fft=1024,
    )
    # amestecăm puțin din original: Whisper e sensibil la artefactele „muzicale” ale denoiser-elor
    return (0.85 * out + 0.15 * audio).astype(np.float32)


def prepare(src: Path, workdir: Path) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Returnează (audio_asr, audio_clean, sr, stats).

    Măsurat pe înregistrarea Medpark: denoise-ul agresiv lasă artefacte care fac detectorul de limbă
    să încline spre rusă (ro 143 s -> 81 s din 240 s). De aceea:
      - audio_asr   = filtre de bandă + normalizare (Whisper e antrenat pe audio zgomotos, îi e mai bine așa)
      - audio_clean = + declick + FFT denoise + spectral gating — pentru VAD și amprentele vocale (diarizare)
    """
    cfg = get_config()
    workdir.mkdir(parents=True, exist_ok=True)
    if not cfg.audio.get("denoise"):
        asr_wav = decode(src, workdir / "asr.wav", clean=True, full=False)
        audio_asr, sr = sf.read(asr_wav, dtype="float32")
        return audio_asr, audio_asr, sr, {"duration_s": round(len(audio_asr) / sr, 2), "sample_rate": sr}
    asr_wav, clean_wav = workdir / "asr.wav", workdir / "clean.wav"
    decode_both(src, asr_wav, clean_wav)
    audio_asr, sr = sf.read(asr_wav, dtype="float32")
    audio_clean, _ = sf.read(clean_wav, dtype="float32")
    stats = {"duration_s": round(len(audio_asr) / sr, 2), "sample_rate": sr}
    n = min(len(audio_asr), len(audio_clean))
    audio_asr, audio_clean = audio_asr[:n], audio_clean[:n]
    stats["noise_floor_db_before"] = _noise_floor_db(audio_asr, sr)
    audio_clean = spectral_denoise(audio_clean, sr, float(cfg.audio.get("denoise_strength", 0.8)))
    stats["noise_floor_db_after"] = _noise_floor_db(audio_clean, sr)
    sf.write(clean_wav, audio_clean, sr, subtype="PCM_16")
    return audio_asr, audio_clean, sr, stats


def _noise_floor_db(audio: np.ndarray, sr: int) -> float:
    frame = sr // 20
    n = len(audio) // frame
    if n == 0:
        return -99.0
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    return round(float(20 * np.log10(np.percentile(rms, 10))), 1)
